#!/usr/bin/env python3
"""
filter_false_negatives.py -- cross-encoder false-negative filter (step 2.5 of 3).

The single highest-value, best-evidenced improvement to hard-negative training in the
retrieval literature (RocketQA, NV-Retriever, et al.): re-score each mined hard negative
with the CROSS-ENCODER (a stronger relevance judge than the bi-encoder that mined it) and
DROP negatives that are actually relevant -- "false negatives". Training against false
negatives produces conflicting gradients that push the model away from genuinely relevant
documents.

Why this matters here specifically:
  * build_dataset.py mines by RANK WINDOW [10,60) -- exactly the band where false negatives
    concentrate (docs the model already ranks highly).
  * The corpus has structural duplication (patent families, cross-source trials), so a mined
    "negative" is often a near-paraphrase of the positive.

Two filters, applied per (query, positive, negatives) triple:
  1. POSITIVE-AWARE THRESHOLD (NV-Retriever "TopK-PercPos"): drop any negative whose
     cross-encoder score >= positive_score * (1 - margin). Scales the bar to how strong the
     positive itself is, rather than a flat cutoff.
  2. ABSOLUTE CEILING: also drop any negative whose cross-encoder score >= abs_max (catches
     cases where the positive itself scored low but a negative is unambiguously relevant).

Input/Output: same schema as build_dataset.py --
    {"query": str, "positive": doc_id, "negatives": [doc_id, ...], ...}
Doc TEXT is not stored; it is joined on doc_id against docs.jsonl (as train.py does).

Usage:
  # inspect drop rate, write nothing:
  python filter_false_negatives.py --device cuda:0 --report-only

  # commit (default in-path = finetune/train_triples.jsonl):
  python filter_false_negatives.py --device cuda:0 \
      --out data/finetune/train_triples_cef.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism
from discovery_hub.schema import read_docs


def _load_triples(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _text_for(doc, max_chars: int) -> str:
    """The same text the embedder/reranker sees for a doc: embedding_text, capped."""
    t = getattr(doc, "embedding_text", None) or ""
    if not t:
        # fall back to title + abstract if embedding_text isn't populated
        t = " ".join(filter(None, [getattr(doc, "title", "") or "",
                                    getattr(doc, "abstract", "") or ""]))
    return t[:max_chars]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=None,
                    help="triples from build_dataset.py "
                         "(default: finetune/train_triples.jsonl)")
    ap.add_argument("--out", default=None,
                    help="filtered triples out (default: finetune/train_triples_cef.jsonl)")
    ap.add_argument("--device", default=None, help="e.g. cuda:0")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="cross-encoder predict batch size")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="positive-aware: drop neg if ce >= pos*(1-margin). NV-Retriever "
                         "uses 0.05 (i.e. the 95%% rule).")
    ap.add_argument("--abs-max", type=float, default=0.9,
                    help="absolute ceiling: drop neg if ce >= abs-max regardless of pos.")
    ap.add_argument("--min-negatives", type=int, default=4,
                    help="report how many triples still have >= this many negatives after "
                         "filtering (a triple below this is weak training signal).")
    ap.add_argument("--drop-empty", action="store_true",
                    help="omit triples left with 0 negatives from the output "
                         "(default: keep them; MNRL still uses in-batch negatives).")
    ap.add_argument("--report-only", action="store_true",
                    help="compute + print stats, write nothing.")
    ap.add_argument("--max-triples", type=int, default=0,
                    help="cap for a quick sample run (0 = all).")
    ap.add_argument("--rerank-max-length", type=int, default=None,
                    help="override cross-encoder max_length (default: config value/512).")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    root = config.DATA_ROOT
    in_path = Path(args.in_path) if args.in_path else root / "finetune" / "train_triples.jsonl"
    out_path = Path(args.out) if args.out else root / "finetune" / "train_triples_cef.jsonl"
    max_len = args.rerank_max_length or getattr(config.RETRIEVAL, "rerank_max_length", 512)

    if not in_path.exists():
        print(f"ERROR: input triples not found: {in_path}", file=sys.stderr)
        return 2

    print(f"==> loading triples: {in_path}")
    triples = _load_triples(in_path)
    if args.max_triples:
        triples = triples[: args.max_triples]
    print(f"    {len(triples):,} triples")

    # doc_id -> text, joined from docs.jsonl (same source train.py uses)
    print(f"==> loading docs.jsonl for text join ...")
    docs = {d.doc_id: d for d in read_docs(config.NORM_DIR / "docs.jsonl")}
    print(f"    {len(docs):,} docs")

    print(f"==> loading cross-encoder {config.RERANK_MODEL} (max_len={max_len}) ...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(config.RERANK_MODEL, max_length=max_len, device=args.device)

    # Build the full (query, doc_text) pair list across all triples in one batched pass:
    # each triple contributes 1 positive pair + len(negatives) negative pairs.
    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, int]] = []  # (triple_idx, -1 for positive else neg_position)
    missing_pos = 0
    for ti, t in enumerate(triples):
        q = t["query"]
        pos = docs.get(t["positive"])
        if pos is None:
            missing_pos += 1
            index.append((ti, -2))       # sentinel: positive text missing
            pairs.append((q, ""))        # placeholder to keep alignment
        else:
            index.append((ti, -1))
            pairs.append((q, _text_for(pos, max_len * 6)))
        for ni, nid in enumerate(t.get("negatives", [])):
            nd = docs.get(nid)
            index.append((ti, ni))
            pairs.append((q, _text_for(nd, max_len * 6) if nd else ""))

    print(f"==> scoring {len(pairs):,} (query,doc) pairs with cross-encoder ...")
    t0 = time.time()
    scores = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    print(f"    scored in {time.time()-t0:,.0f}s")

    # regroup scores per triple
    pos_score: dict[int, float] = {}
    neg_scores: dict[int, list[tuple[int, float]]] = {}
    for (ti, tag), sc in zip(index, scores):
        sc = float(sc)
        if tag == -1:
            pos_score[ti] = sc
        elif tag == -2:
            pos_score[ti] = float("nan")
        else:
            neg_scores.setdefault(ti, []).append((tag, sc))

    # apply the two filters
    total_neg = 0
    dropped_posaware = 0
    dropped_absmax = 0
    kept_neg = 0
    out_rows = []
    hist_kept = {}  # n_negatives_after -> count
    for ti, t in enumerate(triples):
        ps = pos_score.get(ti, float("nan"))
        bar = (ps * (1.0 - args.margin)) if ps == ps else None  # nan check
        survivors = []
        for ni, sc in sorted(neg_scores.get(ti, [])):
            total_neg += 1
            drop = False
            if bar is not None and sc >= bar:
                dropped_posaware += 1
                drop = True
            if sc >= args.abs_max:
                if not drop:
                    dropped_absmax += 1
                drop = True
            if not drop:
                survivors.append(t["negatives"][ni])
        kept_neg += len(survivors)
        hist_kept[len(survivors)] = hist_kept.get(len(survivors), 0) + 1
        new_t = dict(t)
        new_t["negatives"] = survivors
        if survivors or not args.drop_empty:
            out_rows.append(new_t)

    # ---- report ----
    n_at_min = sum(v for k, v in hist_kept.items() if k >= args.min_negatives)
    print("\n================ FALSE-NEGATIVE FILTER REPORT ================")
    print(f"  triples in                : {len(triples):,}")
    if missing_pos:
        print(f"  triples w/ missing positive text : {missing_pos:,} (no positive-aware bar)")
    print(f"  negatives in              : {total_neg:,}")
    print(f"  dropped (positive-aware, ce>=pos*{1-args.margin:.2f}) : {dropped_posaware:,}")
    print(f"  dropped (absolute, ce>={args.abs_max})            : {dropped_absmax:,}")
    print(f"  negatives kept            : {kept_neg:,} "
          f"({100*kept_neg/max(total_neg,1):.1f}%)")
    print(f"  mean negatives/triple     : {kept_neg/max(len(triples),1):.2f} "
          f"(was {total_neg/max(len(triples),1):.2f})")
    print(f"  triples with >= {args.min_negatives} negatives : "
          f"{n_at_min:,} ({100*n_at_min/max(len(triples),1):.1f}%)")
    # small histogram of surviving negative counts
    print("  distribution of surviving negatives/triple:")
    for k in sorted(hist_kept):
        if k <= 8 or k % 2 == 0:
            print(f"     {k:>2} negs : {hist_kept[k]:,}")
    print("=============================================================")

    if args.report_only:
        print("\n[report-only] nothing written. Re-run with --out to commit.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n==> wrote {len(out_rows):,} triples -> {out_path}")
    print(f"    train with:  python -m discovery_finetune.train "
          f"--triples {out_path} --loss mnrl --batch-size 16 --n-negatives 4 --epochs 1")
    print(f"    then re-embed / re-index / eval and compare to 0.406 (0.6B) baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
