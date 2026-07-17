#!/usr/bin/env python3
"""
05_build_index.py  --  LAYER 2, step 5 of 6.

Builds the vector search index from data/embeddings/doc_vectors.npy.

  TARGET: built on Anvil (CPU), shipped to Drew for serving.

We default to a FAISS IndexFlatIP (exact inner-product search). Exact search is
chosen deliberately for the reproducibility story: Flat search is deterministic
and returns identical results every run, whereas approximate (HNSW/IVF) indexes
can return slightly different neighbors. If you switch to an ANN index for scale,
say so in the stability report -- it is a real source of retrieval drift.

If faiss is not installed, we fall back to a pure-numpy brute-force index so the
pipeline still runs (mock/CI).

Outputs:
  data/index/faiss.index           (faiss mode)  OR
  data/index/mock_index.npz        (numpy fallback)

Usage:
  python 05_build_index.py
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from discovery_hub import config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-numpy", action="store_true",
                    help="skip faiss and build the numpy brute-force index")
    args = ap.parse_args()

    config.ensure_dirs()
    vectors = np.load(config.EMB_DIR / "doc_vectors.npy")
    ids = json.loads((config.EMB_DIR / "doc_ids.json").read_text())
    assert vectors.shape[0] == len(ids), "vector/id count mismatch"
    # Counting rows is not enough: duplicate doc_ids pass the count check and then
    # collapse in every doc_id-keyed dict downstream (BM25 below, 07's docs/_docid_to_row,
    # fusion's RRF accumulator). 2,631 SBIR rows vanished from BM25 this way with no error.
    if len(set(ids)) != len(ids):
        from collections import Counter
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise SystemExit(
            f"FATAL: {len(ids) - len(set(ids))} duplicate doc_ids across {len(dupes)} ids "
            f"(e.g. {dupes[:3]}). A doc_id-keyed dict will index the last-wins document "
            "repeatedly and silently drop the rest. Make doc_ids unique before building.")
    dim = vectors.shape[1]
    print(f"Indexing {vectors.shape[0]} vectors of dim {dim}")

    used_faiss = False
    if not args.force_numpy:
        try:
            import faiss

            index = faiss.IndexFlatIP(dim)   # exact, deterministic
            index.add(vectors.astype(np.float32))
            faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
            used_faiss = True
            print(f"  FAISS IndexFlatIP (exact) -> {config.INDEX_DIR / 'faiss.index'}")
        except ImportError:
            print("  faiss not available -> numpy fallback")

    if not used_faiss:
        np.savez(config.INDEX_DIR / "mock_index.npz", vectors=vectors.astype(np.float32))
        print(f"  numpy brute-force index -> {config.INDEX_DIR / 'mock_index.npz'}")

    # Copy doc_ids alongside the index so the serving layer is self-contained.
    shutil.copy(config.EMB_DIR / "doc_ids.json", config.INDEX_DIR / "doc_ids.json")
    print("  copied doc_ids.json into index dir")

    # Build the BM25 keyword index (the lexical half of hybrid retrieval) over the
    # SAME corpus, in doc_id order, and persist it next to the vector index. At
    # production scale this JSON postings file becomes a real search service
    # (OpenSearch / Tantivy); the Retriever interface is unchanged by that swap.
    from discovery_hub.keyword import BM25Index
    from discovery_hub.schema import read_docs
    by_id = {d.doc_id: d for d in read_docs(config.NORM_DIR / "docs.jsonl")}
    ordered = [by_id[i] for i in ids if i in by_id]
    bm25 = BM25Index.build(ordered, k1=config.RETRIEVAL.bm25_k1,
                           b=config.RETRIEVAL.bm25_b)
    bm25.save(config.INDEX_DIR / "bm25.json")
    print(f"  BM25 keyword index ({len(ordered)} docs, "
          f"{len(bm25.idf)} terms) -> {config.INDEX_DIR / 'bm25.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
