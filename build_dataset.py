#!/usr/bin/env python3
"""
build_dataset.py -- hard-negative mining for embedder fine-tuning (step 2 of 3).

    generate_queries.py  ->  build_dataset.py  ->  train.py

Takes the (query, positive-doc) pairs produced by generate_queries.py and mines HARD
NEGATIVES for each: documents the CURRENT embedder ranks highly for the query but that
are not the gold positive. Training against hard negatives (rather than random ones) is
what teaches the model the fine distinctions that matter -- with random negatives the
task is trivial and the model learns nothing about the register gap.

FALSE-NEGATIVE GUARDS (the thing that silently ruins hard-negative mining):
  the corpus is full of near-duplicates -- patent family members, continuations, and
  repeated trial titles. The top-ranked non-positive is very often ALSO relevant, so
  training on it as a negative actively teaches the model wrong things. Two guards:
    --range-min N  skip the N highest-ranked candidates (family members / continuations)
    --range-max M  mine only from ranks [range_min, range_max)
    --max-neg-score S  drop candidates scoring above S (near-identical documents)
  NOTE: negatives are NOT required to score below the positive. Hard negatives are
  precisely the docs the current model ranks ABOVE the gold -- discarding those would
  throw away every query the model currently gets wrong, which are the ones that matter.

Output JSONL, one row per query:
    {"query": ..., "positive": doc_id, "negatives": [doc_id, ...],
     "pos_score": float, "neg_scores": [...], "pos_rank": int|null}
Doc TEXT is not duplicated here -- train.py joins on doc_id against docs.jsonl, which
keeps this file small (ids, not abstracts).

Resumable: mining is sharded and each shard is written atomically, so an interrupted
run resumes (the query embedding + FAISS sweep over 600k docs is not cheap).

  TARGET: Drew (needs the index + vectors). GPU for query embedding, CPU for search.

Usage:
  python build_dataset.py --mock --pairs pairs.jsonl --out triples.jsonl   # plumbing test
  python build_dataset.py \
      --pairs data/finetune/synthetic_queries.jsonl \
      --out   data/finetune/train_triples.jsonl \
      --n-negatives 8 --range-min 10 --range-max 60 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder


def _shard_path(d: Path, i: int) -> Path:
    return d / f"shard_{i:05d}.jsonl"


def _load_pairs(path: Path, max_queries: int = 0) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q, did = r.get("query"), r.get("doc_id")
            if q and did:
                rows.append({"query": q, "positive": did})
            if max_queries and len(rows) >= max_queries:
                break
    return rows


class _Searcher:
    """Top-k inner-product search: FAISS if available, else a chunked numpy matmul."""

    def __init__(self, vectors: np.ndarray, index_path: Path | None):
        self.vectors = vectors
        self.index = None
        if index_path and index_path.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(index_path))
                print(f"  search backend: FAISS ({self.index.ntotal} vectors)")
            except ImportError:
                pass
        if self.index is None:
            print(f"  search backend: numpy matmul ({vectors.shape[0]} vectors)")

    def search(self, qvecs: np.ndarray, k: int):
        if self.index is not None:
            return self.index.search(qvecs.astype(np.float32), k)   # scores, idxs
        sims = qvecs @ self.vectors.T                                # [nq, ndocs]
        idxs = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        part = np.take_along_axis(sims, idxs, axis=1)
        order = np.argsort(-part, axis=1)
        idxs = np.take_along_axis(idxs, order, axis=1)
        scores = np.take_along_axis(part, order, axis=1)
        return scores, idxs


def mine_shard(rows, embedder, searcher, doc_ids, id2row, vectors,
               n_neg: int, range_min: int, range_max: int, max_neg_score: float,
               margin: float, batch_size: int) -> list[dict]:
    """
    Mine negatives from the RANK WINDOW [range_min, range_max).

    Hard negatives are supposed to be docs the current model ranks ABOVE the positive
    -- those are its mistakes, and MNRL's job is to push them down. So we do NOT
    require a negative to score below the positive (that would silently discard every
    query the model currently gets wrong, i.e. exactly the ones we need to train on).

    The false-negative guard is the rank window instead: ranks [0, range_min) are the
    most likely to be genuinely relevant (patent family members, continuations), so we
    skip them; we sample from the window beyond. `max_neg_score` additionally drops
    near-identical documents. `margin` remains available but defaults to OFF.
    """
    import hashlib

    qvecs = embedder.encode_queries([r["query"] for r in rows], batch_size=batch_size)
    qvecs = np.asarray(qvecs, dtype=np.float32)
    k = max(range_max, range_min + n_neg) + 5          # buffer for excluding the positive
    scores, idxs = searcher.search(qvecs, k)

    out = []
    for i, r in enumerate(rows):
        pos_id = r["positive"]
        prow = id2row.get(pos_id)
        if prow is None:                               # positive not in the index
            continue
        pos_score = float(qvecs[i] @ vectors[prow])

        pos_rank, pool, pool_scores = None, [], []
        for rank in range(idxs.shape[1]):
            did = doc_ids[int(idxs[i, rank])]
            sc = float(scores[i, rank])
            if did == pos_id:
                pos_rank = rank
                continue
            if rank < range_min or rank >= range_max:  # outside the mining window
                continue
            if max_neg_score and sc >= max_neg_score:  # near-identical -> presumed dup
                continue
            if margin > 0 and sc >= pos_score - margin:
                continue                               # optional strict guard (off by default)
            pool.append(did)
            pool_scores.append(sc)

        if not pool:
            continue
        # deterministic per-query sample of the window (diversity beats always taking
        # the same top ranks); seeded by query text so resume/reruns are identical.
        seed = int(hashlib.md5(r["query"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        take = min(n_neg, len(pool))
        sel = rng.choice(len(pool), size=take, replace=False)
        sel = sorted(sel, key=lambda j: -pool_scores[j])   # hardest first
        negs = [pool[j] for j in sel]
        nscores = [pool_scores[j] for j in sel]

        out.append({"query": r["query"], "positive": pos_id, "negatives": negs,
                    "pos_score": round(pos_score, 4),
                    "neg_scores": [round(s, 4) for s in nscores],
                    "pos_rank": pos_rank})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=None,
                    help="synthetic_queries.jsonl from generate_queries.py")
    ap.add_argument("--out", default=None, help="output triples JSONL")
    ap.add_argument("--n-negatives", type=int, default=8,
                    help="hard negatives sampled per query")
    ap.add_argument("--range-min", type=int, default=10,
                    help="skip ranks [0,range_min): the most likely genuinely-relevant "
                         "docs (patent family members / continuations)")
    ap.add_argument("--range-max", type=int, default=60,
                    help="mine negatives from ranks [range_min, range_max)")
    ap.add_argument("--max-neg-score", type=float, default=0.95,
                    help="drop candidates scoring above this (near-identical docs)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="OFF by default. If >0, additionally require "
                         "score < pos_score - margin. WARNING: with a weak base model "
                         "this discards every query the model currently gets wrong -- "
                         "exactly the ones worth training on.")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--shard-size", type=int, default=2000, help="queries per checkpoint")
    ap.add_argument("--batch-size", type=int, default=64, help="query-embedding batch")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (real mode)")
    ap.add_argument("--mock", action="store_true", help="mock embedder; no GPU/model")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore existing shards")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    root = config.DATA_ROOT
    pairs_path = Path(args.pairs) if args.pairs else \
        root / "finetune" / "synthetic_queries.jsonl"
    out_path = Path(args.out) if args.out else root / "finetune" / "train_triples.jsonl"
    if not pairs_path.exists():
        print(f"ERROR: pairs not found: {pairs_path}\n  run generate_queries.py first.",
              file=sys.stderr)
        return 2
    vec_path = config.EMB_DIR / "doc_vectors.npy"
    ids_path = (config.INDEX_DIR / "doc_ids.json")
    if not ids_path.exists():
        ids_path = config.EMB_DIR / "doc_ids.json"
    if not (vec_path.exists() and ids_path.exists()):
        print("ERROR: need doc_vectors.npy + doc_ids.json (run stages 04/05).",
              file=sys.stderr)
        return 2

    shards_dir = out_path.parent / (out_path.stem + "_shards")
    if args.restart and shards_dir.exists():
        for p in shards_dir.glob("*.jsonl"):
            p.unlink()
    shards_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_pairs(pairs_path, args.max_queries)
    doc_ids = json.loads(ids_path.read_text())
    id2row = {d: i for i, d in enumerate(doc_ids)}
    vectors = np.load(vec_path, mmap_mode="r")
    print(f"build_dataset: {len(rows)} query pairs | {len(doc_ids)} docs "
          f"| n_neg={args.n_negatives} window=[{args.range_min},{args.range_max}) "
          f"max_neg_score={args.max_neg_score} margin={args.margin or 'off'}")

    embedder = get_embedder(mock=True) if args.mock else \
        get_embedder(mock=False, device=args.device, batch_size=args.batch_size)
    searcher = _Searcher(np.asarray(vectors), config.INDEX_DIR / "faiss.index")

    n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    pending = [i for i in range(n_shards) if not _shard_path(shards_dir, i).exists()]
    print(f"  {n_shards} shard(s); {len(pending)} pending")

    t0 = time.time()
    for n, i in enumerate(pending, 1):
        chunk = rows[i * args.shard_size:(i + 1) * args.shard_size]
        mined = mine_shard(chunk, embedder, searcher, doc_ids, id2row, vectors,
                           args.n_negatives, args.range_min, args.range_max,
                           args.max_neg_score, args.margin, args.batch_size)
        tmp = shards_dir / f"shard_{i:05d}.tmp"
        with tmp.open("w") as fh:
            for r in mined:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, _shard_path(shards_dir, i))
        rate = n / max(time.time() - t0, 1e-6)
        print(f"  shard {i:>4} ({len(chunk)} q -> {len(mined)} triples) "
              f"[{n}/{len(pending)}, {rate*60:.1f} shards/min]", flush=True)

    # ---- combine + stats ----
    triples, pos_ranks, pos_scores, neg_scores = [], [], [], []
    for i in range(n_shards):
        p = _shard_path(shards_dir, i)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            triples.append(line)
            pos_scores.append(r["pos_score"])
            neg_scores.extend(r["neg_scores"])
            if r["pos_rank"] is not None:
                pos_ranks.append(r["pos_rank"])

    with out_path.open("w") as fh:
        for line in triples:
            fh.write(line + "\n")
    print(f"\nwrote {len(triples)} triples -> {out_path}")
    if triples:
        kept = len(triples) / max(len(rows), 1)
        print(f"  kept {kept*100:.1f}% of queries (rest had no negative pass the guards)")
        print(f"  mean pos_score={np.mean(pos_scores):.3f}  "
              f"mean neg_score={np.mean(neg_scores):.3f}  "
              f"mean margin={np.mean(pos_scores) - np.mean(neg_scores):.3f}")
        hit10 = sum(1 for r in pos_ranks if r < 10) / max(len(rows), 1)
        print(f"  positive appeared in top-10 for {hit10*100:.1f}% of queries "
              f"(sanity: should track your dense Recall@10)")

    if not args.keep_shards:
        for i in range(n_shards):
            p = _shard_path(shards_dir, i)
            if p.exists():
                p.unlink()
        try:
            shards_dir.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
