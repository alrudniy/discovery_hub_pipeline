"""
Central configuration for the Discovery Hub pipeline.

This file is the single place that encodes the two scaling profiles (MVP vs FULL),
the two compute targets (Anvil batch HPC vs the Drew always-on server), all on-disk
paths, and the per-source data specs. Every numbered script imports from here so the
pipeline has one source of truth.

COMPUTE PLACEMENT (see the engineering memo for the full rationale):

    Stage                          Target   Why
    -----------------------------  -------  ---------------------------------------
    01 download / 02 parse         Anvil*   I/O + RAM heavy, not GPU heavy
    03 build graph                 Anvil*   RAM heavy (entity resolution)
    04 generate embeddings         Anvil    A100/H100 batch job (millions of docs)
    05 build index                 Anvil    CPU; ships the index to Drew
    06 train R-GCN                 Anvil    single A100 w/ neighbor sampling
    07 retrieve + rank             Drew     always-on retrieval service
    08 multi-agent RAG             Drew     always-on LLM serving (vLLM)
    09 stability harness           either   runs anywhere; pin hardware when measuring

    (*) The MVP biomedical slice is < ~200 GB and fits on Drew, so for the MVP you
        can run 01-06 on Drew too. Full scale (~2-3 TB working set) needs Anvil.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Global knobs
# --------------------------------------------------------------------------- #
SEED = int(os.environ.get("DH_SEED", "20240611"))

# Embedding model. Qwen3-Embedding-0.6B is the production target (fits in 12 GB).
# The mock embedder produces deterministic vectors of EMBED_DIM so the whole
# pipeline runs without a GPU and is bit-reproducible across machines.
EMBED_MODEL = os.environ.get("DH_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM = int(os.environ.get("DH_EMBED_DIM", "1024"))

# Qwen3-Embedding is instruction-aware and ASYMMETRIC: queries are prefixed with
# "Instruct: {task}\nQuery: " while documents are embedded raw (no instruction).
# This task description is what tells the model to bridge the register gap between
# a clinically/commercially phrased research interest (query) and the legal/
# technical language of a patent or trial record (document) -- the core matching
# challenge of this project. Override via DH_QUERY_INSTRUCTION.
QUERY_INSTRUCTION = os.environ.get(
    "DH_QUERY_INSTRUCTION",
    "Given a pharmaceutical or biotech company's research interest, retrieve "
    "patents, clinical trials, and university inventions that are technically "
    "relevant and potentially available for licensing or partnership.")

# Reranker used in the second retrieval stage (real mode only).
RERANK_MODEL = os.environ.get("DH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Self-hosted explanation LLM for Layer 3 (Drew, 12 GB => ~7-8B at 4-bit).
LLM_MODEL = os.environ.get("DH_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# --------------------------------------------------------------------------- #
# Env parsing helpers
# --------------------------------------------------------------------------- #
_FALSEY = {"0", "false", "no", "off", ""}


def env_bool(name: str, default: bool) -> bool:
    """
    Parse a boolean env var. "0"/"false"/"no"/"off"/"" -> False, anything else
    -> True. Case-insensitive. Returns `default` when the var is unset, so every
    caller keeps its current behaviour on a bare environment.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_dir(name: str, default: Path) -> Path:
    """A per-directory override, defaulting to the DATA_ROOT-derived path."""
    raw = os.environ.get(name)
    return default if raw is None else Path(raw).resolve()


# --------------------------------------------------------------------------- #
# Paths -- everything lives under DH_DATA_ROOT (override with an env var so the
# same code points at Drew local disk or Anvil scratch without edits).
#
# WHY the per-directory overrides exist: the artifacts of this project are NOT
# all under one root any more. The merged knowledge graph (1.49M nodes / 4.52M
# edges) and its R-GCN embeddings live under `data_merged/`, while the doc
# embeddings and the FAISS/BM25 index live under `data/`. A single DH_DATA_ROOT
# physically cannot address both, so each directory gets its own override that
# defaults to the DATA_ROOT-derived path. With no env vars set, every path below
# resolves exactly as it did before -- these overrides are purely additive.
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("DH_DATA_ROOT", "./data")).resolve()

RAW_DIR = _env_dir("DH_RAW_DIR", DATA_ROOT / "raw")          # 01 -> raw source records (JSONL)
NORM_DIR = _env_dir("DH_NORM_DIR", DATA_ROOT / "normalized")  # 02 -> DiscoveryDoc records (JSONL)
GRAPH_DIR = _env_dir("DH_GRAPH_DIR", DATA_ROOT / "graph")     # 03 -> nodes/edges.jsonl, graph_meta.json
EMB_DIR = _env_dir("DH_EMB_DIR", DATA_ROOT / "embeddings")    # 04 -> doc_vectors.npy + doc_ids.json
INDEX_DIR = _env_dir("DH_INDEX_DIR", DATA_ROOT / "index")     # 05 -> faiss.index (or mock_index.npz)
ARTIFACT_DIR = _env_dir("DH_ARTIFACT_DIR", DATA_ROOT / "artifacts")  # 06 -> rgcn_node_emb.npy + node_ids.json
REPORT_DIR = _env_dir("DH_REPORT_DIR", DATA_ROOT / "reports")  # 09 -> stability_report.json / .md

ALL_DIRS = [RAW_DIR, NORM_DIR, GRAPH_DIR, EMB_DIR, INDEX_DIR, ARTIFACT_DIR, REPORT_DIR]


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data sources. `mvp_records` is what 01 pulls in --mvp mode; `full_note`
# documents the real full-scale volume so the numbers from the memo stay
# attached to the code.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceSpec:
    name: str
    license: str
    base_url: str
    mvp_records: int
    full_note: str


SOURCES: dict[str, SourceSpec] = {
    "clinicaltrials": SourceSpec(
        name="ClinicalTrials.gov v2",
        license="public domain",
        base_url="https://clinicaltrials.gov/api/v2/studies",
        mvp_records=2000,
        full_note="~590k studies total; small (low single-digit GB).",
    ),
    "sbir": SourceSpec(
        name="SBIR.gov awards",
        license="public domain",
        base_url="https://api.www.sbir.gov/public/api/awards",
        mvp_records=2000,
        full_note="Full bulk file is 290 MB with abstracts; just download it all.",
    ),
    "uspto": SourceSpec(
        name="USPTO / PatentsView",
        license="CC-BY 4.0",
        base_url="https://search.patentsview.org/api/v1/patent/",
        mvp_records=3000,
        full_note="Granted ~100 GB + pre-grant ~26 GB; ~500 GB unzipped full text.",
    ),
    "openalex": SourceSpec(
        name="OpenAlex works",
        license="CC0",
        base_url="https://api.openalex.org/works",
        mvp_records=5000,
        full_note="Snapshot ~330 GB gzip -> ~1.6 TB; ~250M works. Filter to the "
        "~43M medicine+biology subset, or 1-5M for the MVP.",
    ),
    "autm": SourceSpec(
        name="AUTM Innovation Marketplace",
        license="portal terms",
        base_url="",  # no clean bulk API; provided as pre-scraped JSONL by students
        mvp_records=1000,
        full_note="31k+ listings; respect portal terms before redistributing.",
    ),
}

# Biomedical concept filter for OpenAlex (Topic/Concept ids). These are the
# top-level fields used to cut the 250M-work corpus down to the ~43M slice.
OPENALEX_BIOMED_CONCEPTS = {
    "C71924100": "Medicine",
    "C86803240": "Biology",
    "C185592680": "Chemistry",
    "C70721500": "Pharmacology",
}

# --------------------------------------------------------------------------- #
# Retrieval / RAG knobs
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    # Every knob below is env-overridable, with the historical hardcoded value as
    # the default -- an unset environment reproduces the previous behaviour
    # exactly. The overrides exist because the demo and the eval harness need to
    # ship a *measured* configuration (e.g. DH_USE_KEYWORD=0) without editing
    # this file; a config you have to edit to reconfigure is a config that drifts
    # from what was actually measured. Defaults are read at instantiation time,
    # so tests can set the env and construct a fresh RetrievalConfig().
    top_k_recall: int = field(  # first-stage retrieval depth per signal
        default_factory=lambda: env_int("DH_TOP_K_RECALL", 50))
    top_k_rerank: int = field(  # after cross-encoder rerank
        default_factory=lambda: env_int("DH_TOP_K_RERANK", 10))
    min_confidence: float = field(  # Layer-3 policy gate
        default_factory=lambda: env_float("DH_MIN_CONFIDENCE", 0.35))
    # Retrieval is DENSE-ONLY. Both other channels are off, each for its own
    # measured reason. Slice for every number below: the 123 utility-qrels queries
    # that are independently LLM-adjudicated (utility qrels n the 853-query judged
    # set from the PRE-SPLIT label store); relevant = grade >= 2; pool budget 100.
    # Reproduced the deployed pool exactly (RRF = 0.6798) before trusting any of it.
    #
    # KEYWORD OFF. RRF(dense,keyword) evicts more good candidates than keyword adds:
    #   dense top-100 only ............... recall@100 0.7736
    #   RRF(dense,keyword) -> 100 ........ recall@100 0.6798   <- was production
    # RRF threw away 578 relevant docs dense had already found, to seat keyword's
    # 24 unique ones. BM25 is NOT redundant -- it has +0.0520 unique reach, and
    # union(dense,keyword) hits 0.8256 -- but only at a 200-doc pool, i.e. double
    # the reranker bill. At the current 100-doc budget, dense-only wins by +0.0937.
    # Re-open this by raising the pool budget and unioning, NOT by re-enabling RRF.
    use_keyword: bool = field(
        default_factory=lambda: env_bool("DH_USE_KEYWORD", False))
    # GRAPH OFF -- now on direct evidence, not inference. The graph channel's
    # UNIQUE REACH is exactly zero: across the 62 firing queries (668 relevant docs,
    # 6,200 graph candidates) it surfaced 0 relevant documents that dense's top-100
    # missed. Not "few" -- zero. In its full 600,738-doc ranking the median best
    # rank of ANY relevant doc is 195,898, so no depth and no fusion rescues it.
    # Root cause is upstream of the R-GCN: entity_link.link_query fires on junk
    # ("CXCR4 gene therapy approach" -> organization:science approach), so the
    # channel averages embeddings of unrelated entities. The embedding itself is
    # sound (held-out link-prediction residual AUC 0.62 after regressing out
    # degree). Re-open ONLY after build_surface_index precision is measured, not
    # assumed. See for_ppt_two_bear/graph_unique_reach.json.
    use_graph: bool = field(default_factory=lambda: env_bool("DH_USE_GRAPH", False))
    rrf_k: int = 60             # RRF damping constant
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    graph_weight: float = 0.25  # (deprecated: legacy linear blend; RRF is used now)


RETRIEVAL = RetrievalConfig()
