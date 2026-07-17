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

--------------------------------------------------------------------------------
REGISTER-AWARE SEMANTIC CASCADE (opt-in, 2026-07-15)
--------------------------------------------------------------------------------
  DH_REGISTER_AWARE_CASCADE=0|1   default 0 -- OFF is byte-identical to the path above
  DH_ENABLE_GROUPRANK=0|1         default 0 -- requires the cascade; gated, see below

WHY. Pipeline_2 measured, over 55,605 LLM-adjudicated pairs, that this file's reranker
(config.RERANK_MODEL = bge-reranker-v2-m3) correlates +0.468 with query/document WORD
OVERLAP and only +0.211 with actual usefulness. Qwen3-Reranker-8B is +0.371 / +0.410.
At fixed relevance (judged grade 3) BGE's score swings 4.1x on wording alone; Qwen's
swings 1.15x. 76% of this corpus's judged true matches are LOW-overlap, so the deployed
reranker is systematically demoting the majority of the good answers. The canonical case:
"Monoclonal antibody targeting PD-L1" vs a patent titled "Anti-B7-H1 antibodies" -- B7-H1
IS PD-L1 (older name) -- BGE 0.002, Qwen 0.859, LLM grade 3.

WHEN ON, the retrieve path changes to:
  * candidates = UNION of {untouched 0.6B, fine-tuned 8B, deployed dense, graph (if it
    fires), routed exact-identifier} -- deduped, provenance preserved, NO RRF and no
    cross-model score comparison. An uncapped union cannot have lower recall than any
    constituent, which is the one guarantee this design rests on.
  * global BM25 is DROPPED from the union (measured cross-register regression). Lexical
    retrieval survives only as the routed NCT/patent/CAS/compound-code channel, which is
    a no-op unless the query actually contains an identifier.
  * rerank = Qwen3-Reranker-8B, official chat framing, 8K context, pharma instruction,
    ranked on the yes/no LOGIT MARGIN (the probability saturates at the top of the list).

FALLBACK IS ALWAYS DOWNWARD, never sideways:
    GroupRank failure -> Qwen ordering
    Qwen failure      -> existing dense/RRF ordering (this file's current behavior)
    missing vectors   -> existing dense/RRF ordering, with a warning

PREREQUISITE. The cascade needs doc vectors embedded with the UNTOUCHED 0.6B, which do
not exist yet (~45 min to build). Without them the semantic hedge cannot fire and the
flag degrades to the current path. Set:
    DH_CASCADE_VECTORS_BASE_06B / DH_CASCADE_IDS_BASE_06B
    DH_CASCADE_VECTORS_8B       / DH_CASCADE_IDS_8B        (optional)

DETERMINISM CAVEAT (read before running 09 with the flag on). Qwen reranking is a
temperature-free forward pass over a fixed candidate set, so it is reproducible. GroupRank
partitions are seeded. But GroupRank serves via vLLM, whose continuous batching does not
guarantee bitwise-identical logits across runs even at temperature 0 -- so
DH_ENABLE_GROUPRANK=1 may break 09's Jaccard = 1.0 / tau = 1.0 invariant. Validate 09
with the cascade on and GroupRank OFF first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


# --- Register-Aware Semantic Cascade flags (see module docstring) ------------
REGISTER_AWARE_CASCADE = _flag("DH_REGISTER_AWARE_CASCADE")
ENABLE_GROUPRANK = _flag("DH_ENABLE_GROUPRANK")

# Per-channel vectors. Each channel MUST be embedded with its own model -- vectors are
# not interchangeable across models (different dims, different spaces).
CASCADE_MODEL_BASE_06B = os.environ.get("DH_CASCADE_MODEL_BASE_06B",
                                        "Qwen/Qwen3-Embedding-0.6B")
CASCADE_VECTORS_BASE_06B = os.environ.get("DH_CASCADE_VECTORS_BASE_06B", "")
CASCADE_IDS_BASE_06B = os.environ.get("DH_CASCADE_IDS_BASE_06B", "")
CASCADE_MODEL_8B = os.environ.get("DH_CASCADE_MODEL_8B", "")
CASCADE_VECTORS_8B = os.environ.get("DH_CASCADE_VECTORS_8B", "")
CASCADE_IDS_8B = os.environ.get("DH_CASCADE_IDS_8B", "")


class Retriever:
    def __init__(self, mock: bool = True, device: str | None = None,
                 register_aware: bool | None = None,
                 enable_grouprank: bool | None = None):
        self.mock = mock
        self.device = device
        self._cross_encoder = None          # lazy-loaded once, not per query
        self._explainer = None              # lazy; only if explain() is called

        # --- cascade wiring (all lazy; nothing loads unless the flag is on) ---
        self.register_aware = (REGISTER_AWARE_CASCADE if register_aware is None
                               else register_aware)
        self.enable_grouprank = (ENABLE_GROUPRANK if enable_grouprank is None
                                 else enable_grouprank)
        if self.register_aware and mock:
            # Mock mode has no real embeddings, so there is nothing for a semantic
            # reranker to be right about. Silently running the cascade here would make
            # 09's determinism harness measure a different system than production.
            print("[07] cascade disabled: --mock has no real embeddings")
            self.register_aware = False
        if self.enable_grouprank and not self.register_aware:
            print("[07] DH_ENABLE_GROUPRANK ignored: requires DH_REGISTER_AWARE_CASCADE=1")
            self.enable_grouprank = False
        self._cascade = None
        self._ident_index = None
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

    # -- register-aware cascade (opt-in) -------------------------------------
    def _build_cascade(self):
        """Lazily construct the cascade. Returns None if its prerequisites are missing,
        which makes retrieve() fall back to the existing dense/RRF path."""
        if self._cascade is not None:
            return self._cascade or None
        try:
            from dh2.cascade import RegisterAwareCascade
            from dh2.candidate_pool import CH_FT_4B, CH_FT_8B, CH_SEMANTIC
            from dh2.identifiers import build_identifier_index
            from dh2.retriever2 import DenseRetriever
        except ImportError as e:
            print(f"[07] cascade unavailable ({e}); using dense/RRF path")
            self._cascade = False
            return None

        if not (CASCADE_VECTORS_BASE_06B and os.path.exists(CASCADE_VECTORS_BASE_06B)):
            # This is the load-bearing channel. Without it the union is fine-tuned-only:
            # the exact configuration measured to LOSE 0.089 R@10 on low-overlap queries.
            # Running a "cascade" without it would ship the regression under a new name.
            print("[07] cascade DISABLED: no untouched-0.6B vectors "
                  "(set DH_CASCADE_VECTORS_BASE_06B). Embed the corpus with "
                  f"{CASCADE_MODEL_BASE_06B} first (~45 min). Using dense/RRF path.")
            self._cascade = False
            return None

        retrievers = {
            CH_SEMANTIC: DenseRetriever(CASCADE_MODEL_BASE_06B, CASCADE_VECTORS_BASE_06B,
                                        CASCADE_IDS_BASE_06B, device=self.device),
            CH_FT_4B: DenseRetriever(config.EMBED_MODEL if hasattr(config, "EMBED_MODEL")
                                     else os.environ.get("DH_EMBED_MODEL", ""),
                                     str(config.EMB_DIR / "doc_vectors.npy"),
                                     str(config.INDEX_DIR / "doc_ids.json"),
                                     device=self.device),
        }
        if CASCADE_VECTORS_8B and os.path.exists(CASCADE_VECTORS_8B):
            retrievers[CH_FT_8B] = DenseRetriever(CASCADE_MODEL_8B, CASCADE_VECTORS_8B,
                                                  CASCADE_IDS_8B, device=self.device)
        else:
            print("[07] cascade: no 8B vectors; running semantic + deployed dense only")

        if self._ident_index is None:
            self._ident_index = build_identifier_index(self.docs.values())

        self._cascade = RegisterAwareCascade(
            retrievers, self.docs,
            identifier_index=self._ident_index,
            enable_grouprank=self.enable_grouprank)
        print(f"[07] register-aware cascade ACTIVE "
              f"(channels={sorted(retrievers)}, grouprank={self.enable_grouprank})")
        return self._cascade

    def _retrieve_cascade(self, query: str, qvec: np.ndarray, rerank_k: int,
                          use_graph: bool) -> list[dict] | None:
        """Union -> Qwen rerank -> optional GroupRank. None => caller falls back."""
        cas = self._build_cascade()
        if cas is None:
            return None
        try:
            pool = cas.build_candidate_union(query)

            # The graph signal is orthogonal to the register problem (it is structural,
            # not lexical) and it abstains when no entity links, so it can only ADD
            # candidates -- and adding a channel cannot lower the union's recall. BM25 is
            # deliberately NOT added back: it is the measured cross-register regression.
            graph_linked: list[str] = []
            if use_graph and self.graph_emb is not None and self._tech_rows is not None:
                qg, graph_linked = self._graph_query_vector(query)
                if qg is not None:
                    from dh2.candidate_pool import Candidate
                    for rank, (did, sc) in enumerate(
                            self._graph_search(qg, config.RETRIEVAL.top_k_recall), start=1):
                        if did not in pool:
                            pool[did] = Candidate(did)
                        pool[did].add("graph", rank, float(sc))

            ranked = cas.qwen_rerank(query, pool)
            if not any(r.qwen_logit_margin is not None for r in ranked):
                return None                       # Qwen failed -> dense/RRF fallback
            shortlist = ranked[:cas.cfg.qwen_shortlist]
            if self.enable_grouprank:
                shortlist = cas.group_rerank(query, shortlist)
        except Exception as e:                    # noqa: BLE001
            print(f"[07] cascade failed ({e}); falling back to dense/RRF path")
            return None

        cands = []
        for r in shortlist[:rerank_k]:
            cands.append({
                "doc_id": r.doc_id,
                # `score` stays the primary ordering signal downstream. GroupRank's 0-10
                # utility wins when present, else the Qwen margin.
                "score": (r.group_score if r.group_score is not None
                          else r.qwen_logit_margin),
                # 08 reads text_score as a calibrated [0,1] semantic signal, so it must
                # remain the dense cosine -- NOT the logit margin, which is unbounded.
                # Backfilled against the deployed vectors below, exactly as before.
                "text_score": r.scores.get("ft_4b", 0.0),
                "keyword_score": 0.0,             # global BM25 is not in the cascade union
                "graph_score": r.scores.get("graph", 0.0),
                "graph_linked": bool(graph_linked),
                "rerank_score": r.qwen_logit_margin,
                "qwen_logit_margin": r.qwen_logit_margin,
                "qwen_probability": r.qwen_probability,
                "group_score": r.group_score,
                "channels": r.channels,
                "retrieval_ranks": r.ranks,
            })
        return cands

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

        if self.register_aware:
            cands = self._retrieve_cascade(query, qvec, rerank_k, use_graph)
            if cands is not None:
                return self._enrich(cands, qvec)
            # else: every failure path above already logged; fall through unchanged.
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
        return self._enrich(cands, qvec)

    def _enrich(self, cands: list[dict], qvec: np.ndarray) -> list[dict]:
        """Backfill text_score and attach document fields. Shared by both retrieve paths.

        Backfill the true dense cosine for candidates that surfaced via keyword/graph
        (and so were absent from the dense top-k `dense_map`, leaving text_score=0.0).
        text_score is the calibrated [0,1] semantic-relevance signal that downstream
        confidence reads, so every returned candidate must carry its real cosine, not
        0.0 just because dense wasn't the arm that retrieved it. Vectors are unit-norm
        (verified), so dot product == cosine.

        This matters MORE under the cascade: most cascade candidates arrive from the
        untouched-0.6B or 8B channels, whose scores live in different embedding spaces
        and must never be written into text_score. They get 0.0 above and are backfilled
        here against the DEPLOYED vectors, which is the space text_score has always meant.
        """
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

        # Graph EXPLANATION, attached to the results the caller already has.
        #
        # Off by default (DH_GRAPH_EXPLAIN=1 to enable) for one reason: GraphExplainer
        # loads nodes.jsonl + edges.jsonl (~750 MB, several seconds) and most callers of
        # retrieve() -- the evals above all -- want ranked doc_ids and nothing else.
        # It is lazy AND flagged so no eval silently pays for a field it never reads.
        #
        # This CANNOT affect retrieval. It runs after ranking is decided, writes only to
        # c["graph_facts"], and reads no score. The graph retrieval channel is a separate
        # thing and stays off (config.RETRIEVAL.use_graph=False, measured: zero unique
        # reach, -0.033 candidate recall).
        if _flag("DH_GRAPH_EXPLAIN", "0") and cands:
            try:
                from discovery_hub.graph_explain import GraphExplainer
                exp = self.explain(cands)
                # facts_for() reads each relationship onto BOTH endpoints from a
                # single stored pair -- display symmetry without the double count
                # that inflated 4,026 crossovers to 8,852 in the first histogram.
                for c in cands:
                    c["graph_facts"] = GraphExplainer.facts_for(exp, c["doc_id"])
                if cands:
                    cands[0]["graph_space_facts"] = exp["space"]
                    if exp.get("truncated"):
                        cands[0]["graph_truncated"] = exp["truncated"]
            except Exception as e:  # noqa: BLE001
                # Never let an explanation failure take down a retrieval. Say so
                # loudly rather than returning results that silently lack the field.
                print(f"[07] WARNING: graph explanation failed ({type(e).__name__}: {e}); "
                      f"results returned WITHOUT graph_facts", file=sys.stderr)
        return cands

    # -- graph explanation (post-hoc; NOT a retrieval channel) ---------------
    def explain(self, results, *, graph_dir=None) -> dict:
        """Explain a result set you ALREADY have, using the knowledge graph.

        This is the graph's job now. The entity-linked graph RETRIEVAL channel is off
        (config.RETRIEVAL.use_graph=False) because it was measured to have exactly zero
        unique reach: across the 62 firing queries of the judged slice it surfaced no
        relevant document dense's top-100 had missed, and it cost 0.033 candidate recall
        by seating rank-~196k candidates in the pool via RRF. The cause is that ~1 of 853
        judged queries names a real organization, so query->entity linking fires only on
        junk.

        None of that argument reaches this method, and the signature is why: it takes
        doc_ids, not a query. Nothing is linked, no pool is built, no fusion happens,
        and retrieve() never calls this -- so it cannot displace a candidate or change
        a rank. It is strictly additive to what the user already sees.

        Raises rather than returns unverified prose: every fact carries the (src,rel,dst)
        triples it came from and they are re-checked against the graph before returning.
        """
        from discovery_hub.graph_explain import GraphExplainer
        if getattr(self, "_explainer", None) is None:
            self._explainer = GraphExplainer(graph_dir or config.GRAPH_DIR)
        doc_ids = [c["doc_id"] if isinstance(c, dict) else c for c in results]
        exp = self._explainer.explain(doc_ids)
        v = self._explainer.verify(exp)
        if not v["ok"]:
            raise RuntimeError(
                f"graph explanation cited {v['not_in_graph']} edge(s) absent from the "
                f"graph at {graph_dir or config.GRAPH_DIR}. Refusing to return it.")
        exp["verification"] = v
        return exp

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
    ap.add_argument("--cascade", action="store_true",
                    help="force the register-aware cascade on (= DH_REGISTER_AWARE_CASCADE=1)")
    ap.add_argument("--no-cascade", action="store_true",
                    help="force the cascade off regardless of the env flag")
    ap.add_argument("--grouprank", action="store_true",
                    help="enable the GroupRank-32B final pass (requires --cascade)")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    register_aware = None
    if args.cascade:
        register_aware = True
    if args.no_cascade:
        register_aware = False
    r = Retriever(mock=args.mock, register_aware=register_aware,
                  enable_grouprank=True if args.grouprank else None)
    results = r.retrieve(args.query, rerank_k=args.rerank_k,
                         use_keyword=not args.no_keyword,
                         use_graph=not args.no_graph)
    linked = results[0].get("graph_linked") if results else False
    print(f"\nQuery: {args.query}")
    print(f"(path: {'register-aware cascade' if r.register_aware else 'dense/RRF + BGE'})")
    print(f"(graph signal: {'fired' if linked else 'abstained -- no entity linked'})")
    print(f"Top {len(results)} candidates:\n")
    for i, c in enumerate(results, 1):
        print(f"{i:2d}. [{c['score']:.4f}] {c.get('title','')[:68]}")
        line = (f"     {c['doc_id']}  (text={c['text_score']:.3f} "
                f"bm25={c['keyword_score']:.2f} graph={c['graph_score']:.3f})")
        if c.get("qwen_logit_margin") is not None:
            line += f"\n     qwen_margin={c['qwen_logit_margin']:+.3f}"
            if c.get("group_score") is not None:
                line += f" group={c['group_score']:.1f}"
            line += f"  via={'+'.join(c.get('channels', []))}"
        print(f"{line}  {c.get('source_url','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
