#!/usr/bin/env python3
"""
filter_false_negatives.py -- cross-encoder false-negative filtering for the mined triples.

    generate_queries.py -> build_dataset.py -> [filter_false_negatives.py] -> train.py
                                                         ^ this script (additive, optional)

WHY THIS EXISTS
build_dataset.py mines hard negatives from the rank window [10, 60): the band where the
CURRENT bi-encoder already ranks documents highly. That is exactly where genuinely-
relevant-but-unlabeled documents concentrate -- FALSE NEGATIVES. With a corpus full of
patent families, continuations, and the same trial repeated across sources, a real
fraction of the mined "negatives" are actually relevant. MNRL then pushes the model AWAY
from relevant content on those rows (conflicting gradients), capping how far fine-tuning
can go.

The bi-encoder cannot catch this on its own -- it mined those negatives BECAUSE it scores
them highly. A CROSS-ENCODER sees full query-document interaction the bi-encoder cannot,
so it catches false negatives the embedding-similarity guard (build_dataset's
--max-neg-score) misses. This is the single most-cited fix in the hard-negative
literature (RocketQA denoising; NV-Retriever positive-aware thresholds).

WHAT IT DOES  (per mined row: query, positive, [negatives])
  1. score CE(query, positive)          -> ce_pos   (sigmoid-normalized to [0, 1])
  2. score CE(query, each negative)     -> ce_neg
  3. DROP a negative when it looks relevant relative to the positive:
        ce_neg >= ce_pos * --ce-threshold        (positive-aware, NV-Retriever style; 0.95)
     and, if --ce-abs-floor > 0, ALSO drop any negative with:
        ce_neg >= --ce-abs-floor                 (clearly relevant regardless of positive)
  4. rows left with 0 negatives are dropped; the rest are rewritten with dead negs removed
     (each row keeps its surviving negatives, hardest-CE first).

This is MODEL-AGNOSTIC cleaning: it does not matter which embedder MINED the negatives, so
the cleaned triples train any model (0.6B / 4B / 8B) equally. Doc text is NOT stored in the
triples file -- exactly like train.py, we join on doc_id against docs.jsonl to get text.

  TARGET: any GPU box with the BGE cross-encoder (the same reranker stage 07 uses). ~600k
          (query, doc) pairs for the 66k-triple set; a few minutes on an H100. Run under
          `nohup ... &` so an SSH drop does not kill it (footgun #6). Sharded + resumable.

Usage:
  # 1) inspect first -- scores + would-drop stats, writes NOTHING:
  python filter_false_negatives.py --report-only --device cuda:0

  # 2) then commit the cleaned set:
  python filter_false_negatives.py \
      --triples data/finetune/train_triples.jsonl \
      --docs    data/normalized/docs.jsonl \
      --out     data/finetune/train_triples_cefiltered.jsonl \
      --ce-threshold 0.95 --batch-size 128 --device cuda:0

  python filter_false_negatives.py --mock --triples t.jsonl --docs d.jsonl --out o.jsonl
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

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"   # the reranker stage 07 already uses


# ------------------------------------------------------------------ helpers
def _shard_path(d: Path, i: int) -> Path:
    return d / f"shard_{i:05d}.jsonl"


def load_doc_texts(docs_path: Path) -> dict[str, str]:
    """doc_id -> embedding_text (the exact string stage 04 embeds), matching train.py."""
    out: dict[str, str] = {}
    with docs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("embedding_text") or \
                f"{d.get('title', '')} {d.get('abstract', '')}".strip()
            if d.get("doc_id") and t:
                out[d["doc_id"]] = t
    return out


def load_triples(path: Path, max_rows: int = 0) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("query") and r.get("positive") and r.get("negatives"):
                rows.append(r)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


class _Scorer:
    """Batched cross-encoder scoring. Loads the model ONCE (footgun #1: per-pair load OOMs).
    Returns sigmoid-normalized probabilities in [0, 1] for a list of [query, doc] pairs."""

    def __init__(self, model_name: str, max_length: int, batch_size: int,
                 device: str | None, mock: bool):
        self.batch_size = batch_size
        self.mock = mock
        if mock:
            self.ce = None
            print("  scorer: MOCK (deterministic, no model)")
            return
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model_name, max_length=max_length, device=device)
        print(f"  scorer: CrossEncoder({model_name}) max_length={max_length} "
              f"device={device or 'auto'}")

    def score(self, pairs: list[list[str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        if self.mock:
            # deterministic pseudo-scores from the text so plumbing/tests are reproducible
            import hashlib
            return np.array(
                [int(hashlib.md5((q + "\x00" + d).encode()).hexdigest()[:6], 16) / 0xFFFFFF
                 for q, d in pairs], dtype=np.float32)
        logits = self.ce.predict(pairs, batch_size=self.batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:                       # 2-label head -> softmax, take positive
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return (e[:, 1] / e.sum(axis=1))
        return 1.0 / (1.0 + np.exp(-logits))       # 1-label head (BGE v2-m3) -> sigmoid


def filter_shard(rows, doc_texts, scorer, ce_threshold, ce_abs_floor):
    """Score one shard's positives + negatives, drop false-negative negatives, return
    (cleaned_rows, stats). Rows whose positive text is missing are dropped; negatives
    whose text is missing are dropped."""
    # build a flat pair list with an index map back to (row, kind, neg_idx)
    pairs, idx = [], []
    for ri, r in enumerate(rows):
        pos_text = doc_texts.get(r["positive"])
        if not pos_text:
            continue
        idx.append((ri, "pos", -1)); pairs.append([r["query"], pos_text])
        for ni, nid in enumerate(r["negatives"]):
            nt = doc_texts.get(nid)
            if nt is None:
                continue
            idx.append((ri, "neg", ni)); pairs.append([r["query"], nt])

    probs = scorer.score(pairs)

    ce_pos: dict[int, float] = {}
    ce_neg: dict[int, dict[int, float]] = {}
    for (ri, kind, ni), p in zip(idx, probs):
        if kind == "pos":
            ce_pos[ri] = float(p)
        else:
            ce_neg.setdefault(ri, {})[ni] = float(p)

    out, st = [], {"rows_in": len(rows), "rows_out": 0, "rows_no_pos_text": 0,
                   "rows_emptied": 0, "neg_in": 0, "neg_dropped": 0, "neg_kept": 0,
                   "pos_probs": [], "kept_probs": [], "dropped_probs": []}
    for ri, r in enumerate(rows):
        if ri not in ce_pos:
            st["rows_no_pos_text"] += 1
            continue
        p_pos = ce_pos[ri]
        st["pos_probs"].append(p_pos)
        rel_thr = p_pos * ce_threshold
        keep_negs, keep_scores = [], []
        for ni, nid in enumerate(r["negatives"]):
            p_neg = ce_neg.get(ri, {}).get(ni)
            if p_neg is None:                      # no text for this negative -> drop
                continue
            st["neg_in"] += 1
            is_fn = (p_neg >= rel_thr) or (ce_abs_floor > 0 and p_neg >= ce_abs_floor)
            if is_fn:
                st["neg_dropped"] += 1
                st["dropped_probs"].append(p_neg)
            else:
                keep_negs.append(nid); keep_scores.append(p_neg)
                st["neg_kept"] += 1
                st["kept_probs"].append(p_neg)
        if not keep_negs:
            st["rows_emptied"] += 1
            continue
        order = sorted(range(len(keep_negs)), key=lambda j: -keep_scores[j])  # hardest first
        r_out = dict(r)
        r_out["negatives"] = [keep_negs[j] for j in order]
        r_out["ce_neg_probs"] = [round(keep_scores[j], 4) for j in order]
        r_out["ce_pos_prob"] = round(p_pos, 4)
        out.append(r_out)
        st["rows_out"] += 1
    return out, st


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    dr = os.environ.get("DH_DATA_ROOT", "data")
    ap.add_argument("--triples", default=f"{dr}/finetune/train_triples.jsonl")
    ap.add_argument("--docs", default=f"{dr}/normalized/docs.jsonl")
    ap.add_argument("--out", default=f"{dr}/finetune/train_triples_cefiltered.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="cross-encoder (BGE reranker)")
    ap.add_argument("--ce-threshold", type=float, default=0.95,
                    help="positive-aware: drop a negative if ce_neg >= ce_pos * THIS. "
                         "0.95 = NV-Retriever's 95%% margin. Lower = stricter (drops more).")
    ap.add_argument("--ce-abs-floor", type=float, default=0.0,
                    help="also drop any negative with ce_neg >= THIS absolute prob "
                         "(0 = off). Use to catch clearly-relevant docs when the positive "
                         "itself scores low (genuine register-gap pairs).")
    ap.add_argument("--max-length", type=int, default=512,
                    help="MUST match stage 07's rerank_max_length for comparable scores")
    ap.add_argument("--batch-size", type=int, default=128, help="CE scoring batch")
    ap.add_argument("--shard-size", type=int, default=2000, help="rows per checkpoint")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all triples")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu")
    ap.add_argument("--report-only", action="store_true",
                    help="score + print stats, write NOTHING (inspect before committing)")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore existing shards")
    ap.add_argument("--mock", action="store_true", help="deterministic, no GPU/model")
    args = ap.parse_args()

    triples_path, docs_path, out_path = Path(args.triples), Path(args.docs), Path(args.out)
    for p, what in [(triples_path, "triples"), (docs_path, "docs")]:
        if not p.exists():
            print(f"ERROR: {what} not found: {p}", file=sys.stderr)
            return 2

    rows = load_triples(triples_path, args.max_rows)
    print(f"filter_false_negatives: {len(rows)} triples | loading doc texts ...")
    doc_texts = load_doc_texts(docs_path)
    print(f"  {len(doc_texts)} docs | ce_threshold={args.ce_threshold} "
          f"ce_abs_floor={args.ce_abs_floor or 'off'} "
          f"{'[REPORT-ONLY]' if args.report_only else ''}")

    scorer = _Scorer(args.model, args.max_length, args.batch_size, args.device, args.mock)

    shards_dir = out_path.parent / (out_path.stem + "_cef_shards")
    if args.restart and shards_dir.exists():
        for p in shards_dir.glob("*.jsonl"):
            p.unlink()
    shards_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    pending = [i for i in range(n_shards) if not _shard_path(shards_dir, i).exists()]
    print(f"  {n_shards} shard(s); {len(pending)} pending "
          f"(shard_size={args.shard_size}, batch={args.batch_size})")

    agg = {"rows_in": 0, "rows_out": 0, "rows_no_pos_text": 0, "rows_emptied": 0,
           "neg_in": 0, "neg_dropped": 0, "neg_kept": 0,
           "pos_probs": [], "kept_probs": [], "dropped_probs": []}
    t0 = time.time()
    for n, i in enumerate(pending, 1):
        chunk = rows[i * args.shard_size:(i + 1) * args.shard_size]
        cleaned, st = filter_shard(chunk, doc_texts, scorer,
                                   args.ce_threshold, args.ce_abs_floor)
        tmp = shards_dir / f"shard_{i:05d}.tmp"
        with tmp.open("w") as fh:
            for r in cleaned:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, _shard_path(shards_dir, i))
        rate = n / max(time.time() - t0, 1e-6)
        print(f"  shard {i:>4} ({st['rows_in']} rows -> {st['rows_out']} kept, "
              f"{st['neg_dropped']}/{st['neg_in']} negs dropped) "
              f"[{n}/{len(pending)}, {rate*60:.1f} shards/min]", flush=True)

    # ---- combine surviving shards + aggregate stats ----
    kept_lines, neg_counts = [], []
    for i in range(n_shards):
        p = _shard_path(shards_dir, i)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            kept_lines.append(line)
            neg_counts.append(len(r["negatives"]))
            agg["kept_probs"].extend(r.get("ce_neg_probs", []))
            agg["pos_probs"].append(r.get("ce_pos_prob", 0.0))

    # recompute drop totals across all shards from the raw inputs (report-only safe)
    agg["rows_in"] = len(rows)
    agg["rows_out"] = len(kept_lines)

    if not args.report_only:
        with out_path.open("w") as fh:
            for line in kept_lines:
                fh.write(line + "\n")
        print(f"\nwrote {len(kept_lines)} cleaned triples -> {out_path}")
    else:
        print(f"\n[REPORT-ONLY] would write {len(kept_lines)} cleaned triples "
              f"(nothing written). Re-run without --report-only to commit.")

    # ---- summary ----
    if kept_lines:
        nc = np.array(neg_counts)
        surv4 = int((nc >= 4).sum()); surv2 = int((nc >= 2).sum())
        print(f"  rows: {agg['rows_in']} -> {agg['rows_out']} kept "
              f"({100*agg['rows_out']/max(agg['rows_in'],1):.1f}%)")
        print(f"  usable at n_negatives: >=2 -> {surv2} rows | >=4 -> {surv4} rows "
              f"(train.py drops rows with fewer than --n-negatives negs)")
        print(f"  surviving negatives/row: mean={nc.mean():.2f} min={nc.min()} "
              f"max={nc.max()}")
        if agg["pos_probs"]:
            print(f"  CE prob  positives (kept rows): mean={np.mean(agg['pos_probs']):.3f}")
        if agg["kept_probs"]:
            print(f"  CE prob  negatives KEPT       : mean={np.mean(agg['kept_probs']):.3f}")
        print("  (sanity: kept-negative CE prob should sit well BELOW positive CE prob; "
              "if they overlap, lower --ce-threshold)")

    if not args.keep_shards and not args.report_only:
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
