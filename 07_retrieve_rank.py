#!/usr/bin/env python3
"""
07_retrieve_rank.py  --  LAYER 2 serving (and the seam to Layer 3).

The always-on retrieval service. HYBRID retrieve-then-rerank with three signals
fused by Reciprocal Rank Fusion, then a cross-encoder rerank:

  dense    : query embedding -> top_k by text cosine (FAISS exact)
  keyword  : BM25 over the same corpus (exact terms/acronyms the encoder misses)
  graph    : entity-linked R-GCN signal -- if the query names a graph entity
             (org / inventor / facility), anchor on it and surface structurally
             connected technologies. ABSTAINS when nothing links (no noise).
  fusion   : RRF over whichever signals fired (scale-free; a missing signal just
             isn't fused) -> candidate pool
  rerank   : cross-encoder (real) or fused-score sort (mock) -> top_k_rerank,
             each candidate carrying supporting evidence for Layer 3.

  TARGET: Drew (always-on). Every signal is deterministic (FAISS Flat, BM25, and
          a fixed graph dot product), so the retrieved set and its order are
          reproducible run-to-run -- which 09 verifies (Jaccard = 1.0, tau = 1.0).

Exposes a `Retriever` class imported by 08_multiagent_rag.py and 09_stability_harness.py.

Usage:
  python 07_retrieve_rank.py --mock --query "GLP-1 agonist for metabolic disease"
  python 07_retrieve_rank.py --mock --query "..." --no-keyword --no-graph
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Bound GPU memory fragmentation on small (12 GB) cards -- the reranker/embedder
# "reserved but unallocated" OOM. Set before torch/CUDA initializes; harmless else.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config, entity_link
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder
from discovery_hub.fusion import fuse_to_pool
from discovery_hub.keyword import BM25Index
from discovery_hub.schema import read_docs


class Retriever:
    def __init__(self, mock: bool = True, device: str | None = None):
        self.mock = mock
        self.device = device
        self._cross_encoder = None          # lazy-loaded once, not per query
        self.embedder = get_embedder(mock=mock, device=device)
        self.doc_vectors = np.load(config.EMB_DIR / "doc_vectors.npy")
        self.doc_ids = json.loads((config.INDEX_DIR / "doc_ids.json").read_text())
        # doc_id -> row in doc_vectors, so any candidate's true dense cosine can be read
        # back (used to backfill text_score for keyword/graph-sourced candidates).
        self._docid_to_row = {did: i for i, did in enumerate(self.doc_ids)}
        self.docs = {d.doc_id: d for d in read_docs(config.NORM_DIR / "docs.jsonl")}

        # Dense index (FAISS exact if available, else numpy brute force).
        self._faiss = None
        faiss_path = config.INDEX_DIR / "faiss.index"
        if faiss_path.exists():
            try:
                import faiss
                self._faiss = faiss.read_index(str(faiss_path))
            except ImportError:
                self._faiss = None

        # BM25 keyword index: load the one built by stage 05, else build from docs
        # (defensive so 07 works even if 05 wasn't re-run).
        bm25_path = config.INDEX_DIR / "bm25.json"
        if bm25_path.exists():
            self.bm25 = BM25Index.load(bm25_path)
        else:
            ordered = [self.docs[i] for i in self.doc_ids if i in self.docs]
            self.bm25 = BM25Index.build(ordered, k1=config.RETRIEVAL.bm25_k1,
                                        b=config.RETRIEVAL.bm25_b)

        # Graph: R-GCN node embeddings + node bookkeeping for the entity-linked
        # query signal. We need (a) row of every node, (b) a surface-form index of
        # entity nodes for linking, (c) the technology rows aligned to doc_ids so
        # the graph can be searched.
        self.graph_emb = None
        self.node_grow: dict[str, int] = {}      # node_id -> row
        self.surface_index: dict[str, set[str]] = {}
        self.docid_to_grow: dict[str, int] = {}   # doc_id -> tech node row
        self._tech_rows = None
        self._tech_docids: list[str] = []
        emb_path = config.ARTIFACT_DIR / "rgcn_node_emb.npy"
        if emb_path.exists():
            self.graph_emb = np.load(emb_path)
            node_ids = json.loads((config.ARTIFACT_DIR / "node_ids.json").read_text())
            self.node_grow = {nid: i for i, nid in enumerate(node_ids)}
            nodes = [json.loads(l) for l in
                     (config.GRAPH_DIR / "nodes.jsonl").read_text().splitlines() if l]
            self.surface_index = entity_link.build_surface_index(nodes)
            for nd in nodes:
                did = nd.get("doc_id")
                if did and nd["node_id"] in self.node_grow:
                    self.docid_to_grow[did] = self.node_grow[nd["node_id"]]
            if self.docid_to_grow:
                self._tech_docids = list(self.docid_to_grow.keys())
                self._tech_rows = np.array(
                    [self.docid_to_grow[d] for d in self._tech_docids], dtype=np.int64)

    # -- individual signals --------------------------------------------------
    def _text_search(self, qvec: np.ndarray, k: int):
        if self._faiss is not None:
            scores, idxs = self._faiss.search(qvec.reshape(1, -1).astype(np.float32), k)
            return idxs[0], scores[0]
        sims = self.doc_vectors @ qvec
        idxs = np.argsort(-sims)[:k]
        return idxs, sims[idxs]

    def _keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.score(query, top_k=k)

    def _graph_query_vector(self, query: str):
        """Build a graph-space query vector from the query's named entities.
        Returns (unit_vector | None, linked_node_ids). None == abstain."""
        linked = entity_link.link_query(query, self.surface_index)
        rows = [self.node_grow[nid] for nid in linked if nid in self.node_grow]
        if not rows:
            return None, linked
        v = self.graph_emb[rows].mean(0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32), linked

    def _graph_search(self, qgvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Rank technologies by structural proximity to the query's entities.
        O(num_docs) dot product -- fine at MVP scale; at production scale this
        wants an ANN index over the graph embeddings."""
        scores = self.graph_emb[self._tech_rows] @ qgvec
        order = sorted(range(len(scores)),
                       key=lambda j: (-scores[j], self._tech_docids[j]))[:k]
        return [(self._tech_docids[j], float(scores[j])) for j in order]

    # -- hybrid retrieve -----------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None,
                 rerank_k: int | None = None, *,
                 use_keyword: bool | None = None,
                 use_graph: bool | None = None) -> list[dict]:
        top_k = top_k or config.RETRIEVAL.top_k_recall
        rerank_k = rerank_k or config.RETRIEVAL.top_k_rerank
        use_keyword = config.RETRIEVAL.use_keyword if use_keyword is None else use_keyword
        use_graph = config.RETRIEVAL.use_graph if use_graph is None else use_graph

        qvec = self.embedder.encode_queries([query])[0]
        lists: dict[str, list[str]] = {}

        d_idx, d_sc = self._text_search(qvec, top_k)
        lists["dense"] = [self.doc_ids[i] for i in d_idx]
        dense_map = {self.doc_ids[i]: float(d_sc[r]) for r, i in enumerate(d_idx)}

        kw_map: dict[str, float] = {}
        if use_keyword and self.bm25 is not None:
            kw = self._keyword_search(query, top_k)
            lists["keyword"] = [d for d, _ in kw]
            kw_map = dict(kw)

        graph_map: dict[str, float] = {}
        linked: list[str] = []
        if use_graph and self.graph_emb is not None and self._tech_rows is not None:
            qg, linked = self._graph_query_vector(query)
            if qg is not None:
                gl = self._graph_search(qg, top_k)
                lists["graph"] = [d for d, _ in gl]
                graph_map = dict(gl)
            # else: graph abstains (its list is simply absent from the fusion)

        pool = fuse_to_pool(lists, top_k, k=config.RETRIEVAL.rrf_k)
        cands = [{"doc_id": did, "score": fscore,
                  "text_score": dense_map.get(did, 0.0),
                  "keyword_score": kw_map.get(did, 0.0),
                  "graph_score": graph_map.get(did, 0.0),
                  "graph_linked": bool(linked)}
                 for did, fscore in pool]

        cands = self._rerank(query, cands)[:rerank_k]
        # Backfill the true dense cosine for candidates that surfaced via keyword/graph
        # (and so were absent from the dense top-k `dense_map`, leaving text_score=0.0).
        # text_score is the calibrated [0,1] semantic-relevance signal that downstream
        # confidence reads, so every returned candidate must carry its real cosine, not
        # 0.0 just because dense wasn't the arm that retrieved it. Vectors are unit-norm
        # (verified), so dot product == cosine.
        for c in cands:
            if not c.get("text_score"):
                row = self._docid_to_row.get(c["doc_id"])
                if row is not None:
                    c["text_score"] = float(self.doc_vectors[row] @ qvec)
        for c in cands:
            d = self.docs.get(c["doc_id"])
            if d:
                c["title"] = d.title
                c["abstract"] = d.abstract
                c["source"] = d.source
                c["source_url"] = d.source_url
                c["organizations"] = d.organizations
        return cands

    def _get_cross_encoder(self):
        """Load the cross-encoder ONCE and cache it. Previously it was constructed
        per query, reloading ~560M weights every call (slow + fragmented GPU mem).
        max_length caps input so a long patent/trial abstract can't OOM a batch."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            max_len = getattr(config.RETRIEVAL, "rerank_max_length", 512)
            self._cross_encoder = CrossEncoder(
                config.RERANK_MODEL, max_length=max_len, device=self.device)
        return self._cross_encoder

    def _rerank(self, query: str, cands: list[dict]) -> list[dict]:
        if self.mock:
            # Deterministic: fused score, ties broken by doc_id.
            return sorted(cands, key=lambda c: (-c["score"], c["doc_id"]))
        ce = self._get_cross_encoder()
        batch_size = getattr(config.RETRIEVAL, "rerank_batch_size", 16)
        pairs = [(query, self.docs[c["doc_id"]].embedding_text) for c in cands]
        rr = ce.predict(pairs, batch_size=batch_size)
        for c, s in zip(cands, rr):
            c["rerank_score"] = float(s)
        return sorted(cands, key=lambda c: (-c["rerank_score"], c["doc_id"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--query", required=True)
    ap.add_argument("--rerank-k", type=int, default=config.RETRIEVAL.top_k_rerank)
    ap.add_argument("--no-keyword", action="store_true", help="disable BM25 half")
    ap.add_argument("--no-graph", action="store_true", help="disable graph signal")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    r = Retriever(mock=args.mock)
    results = r.retrieve(args.query, rerank_k=args.rerank_k,
                         use_keyword=not args.no_keyword,
                         use_graph=not args.no_graph)
    linked = results[0].get("graph_linked") if results else False
    print(f"\nQuery: {args.query}")
    print(f"(graph signal: {'fired' if linked else 'abstained -- no entity linked'})")
    print(f"Top {len(results)} candidates:\n")
    for i, c in enumerate(results, 1):
        print(f"{i:2d}. [{c['score']:.4f}] {c.get('title','')[:68]}")
        print(f"     {c['doc_id']}  (text={c['text_score']:.3f} "
              f"bm25={c['keyword_score']:.2f} graph={c['graph_score']:.3f})  "
              f"{c.get('source_url','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
