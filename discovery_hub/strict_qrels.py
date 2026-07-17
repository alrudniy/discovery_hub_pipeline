"""
THE query set for the strict-RAG measurement harnesses (strict_rag_spec.md 4.1, 4.4).

Pure data loading. It constructs nothing heavy, imports no agent, loads no model,
and never builds a Retriever. Everything here is a file read and a sort.

WHY THIS EXISTS AS ITS OWN MODULE
    The counterfactual harness (4.1) and the citation audit (4.4) must run on the
    SAME queries with the SAME gold evidence, or their numbers cannot be compared
    to each other or to anything in config.py. Two harnesses each re-deriving "the
    123" from a 147 MB label store and a 1.5 GB corpus is two chances to derive it
    differently and never notice. So the slice is derived ONCE, here.

WHAT THIS IS NOT
    It is NOT a retrieval evaluation and it makes no retrieval claim. The doc_ids
    it hands out are GOLD -- they come from the qrels answer key, not from a
    retriever. A harness fed these candidates is measuring GROUNDING (does the
    model stick to the evidence it was given?), NOT recall, NOT ranking, and NOT
    end-to-end system quality. That is deliberate, and it is a better control than
    real retrieval would be: gold evidence removes retrieval quality as a confound
    from a test that is about grounding. It also means no number produced on top of
    this module may be described as an end-to-end result.

WHY GOLD EVIDENCE AND NOT A REAL RETRIEVER (the operational reason too)
    A real Retriever(mock=False) load is ~20.2 GB anon-RSS on a 30 GB box
    (CONFIRMED -- an OOM kill in dmesg). Only ONE fits at a time, and foreign
    explain_demo.py runs from for_ppt_two_bear have already OOM-killed one probe.
    A harness that quietly constructs a second one dies mid-measurement.

THE SLICE, AND EVERY NUMBER IN IT (all reproduced by running this module's own
functions against the real files -- see the module's provenance note below)
    judged_query_ids(DEFAULT_LABEL_STORE) ................. 853 distinct query_id
    utility qrels ......................................... 440 held-out queries
    853 INTERSECT 440 ..................................... 123  <- adjudicated_slice
    The 123 are exactly the "123 utility-qrels queries that are independently
    LLM-adjudicated" that discovery_hub/config.py quotes EVERY retrieval number
    against (recall@100 0.7736 dense-only, 0.6798 RRF, etc). Same slice or the
    numbers are not comparable.

    relevant = grade >= 2. The utility qrels grade 0-3:
        3 = directly_actionable, 2 = clearly_relevant, 1 = tangential, 0 = irrelevant

    ALL 123 are in held_out_query_ids.json -- none of them trained the deployed
    embedder. The slice is clean by construction, not by hope.

WHY select() DEFAULTS TO min_rel=2 AND NOT min_rel=3 (a sizing fact, not taste)
    Measured over the 123: median relevant-docs/query is 1. 63 of the 123 have
    exactly ONE graded relevant doc; only 45 have >= 3. So min_rel=3 caps the pool
    at 45 and CANNOT reach 50 -- asking for 50 at min_rel=3 silently returns 45 and
    every per-query rate then has a different denominator than the one reported.
    min_rel=2 yields 60 candidates; taking the 50 best-covered lands exactly 50.

SCOPE CAVEAT -- READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SLICE
    The corpus is USPTO + clinicaltrials + openalex + SBIR, so it drifts outside
    pharma, and sorting by coverage AMPLIFIES that drift: the single best-covered
    query in the 50 is "permanent hair coloring delivery system" (q-0000038, 115
    relevant docs) -- cosmetics, not pharma. It is NOT dropped. Dropping it would
    be hand-picking the slice to flatter a pharma story, which is the exact move
    this repo exists to not make. Every doc in the selected 50 happens to be
    source-prefixed "uspto:" (measured: 1,772/1,772), so this slice exercises
    patent language only -- it says NOTHING about trial or paper text.

WHAT THIS SLICE CANNOT MEASURE -- THE CONFIDENCE GATE NEVER FIRES
    as_candidates() deliberately emits NO rerank_score (see its comment). Because
    agents/synthesis.py decides reranked-vs-mock from ("rerank_score" in kept[0]),
    a harness built on these candidates SKIPS the 0.35 confidence gate entirely and
    payload["confidence_gate"] will read "NOT APPLIED: no reranker ran...". THE
    0.35 GATE IS THEREFORE UNMEASURED BY ANYTHING BUILT ON THIS MODULE, and any
    report that uses it MUST say so. Fabricating a rerank_score to make the gate
    fire would fabricate a measurement.

PROVENANCE OF THE NUMBERS ABOVE
    Reproduce them (1.1 s measured; reads the 147 MB store + 119 KB qrels, and
    NOT the corpus):
        venv/bin/python -c "from discovery_hub.strict_qrels import *; \
            print(len(judged_query_ids()), len(adjudicated_slice()), len(select()))"
        -> 853 123 50
    Evidence resolution is a separate streaming pass over the 1.5 GB corpus:
    load_evidence(select()) -> 1,772/1,772 doc_ids resolved, 0 with an empty
    abstract, 5.3 s, PEAK RSS 32 MB (measured, this box).

    On 1,772 vs 1,776, since both numbers are true and they get confused: the 50
    queries reference 1,776 (query, doc) evidence pairs but only 1,772 DISTINCT
    doc_ids -- 4 docs are graded relevant for more than one query. load_evidence
    is keyed by doc_id, so it returns 1,772; a per-query count sums to 1,776.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .evaluate import EvalQuery

# --------------------------------------------------------------------------- #
# Paths. Env-override idiom copied from config.py's _env_dir: an unset
# environment reproduces the MEASURED configuration exactly, and the override
# exists so a harness can point at a fixture without editing this file.
# --------------------------------------------------------------------------- #


def _env_path(name: str, default: Path) -> Path:
    """A per-file override, defaulting to the measured path. See config.py::_env_dir."""
    raw = os.environ.get(name)
    return default if raw is None else Path(raw).resolve()


# --------------------------------------------------------------------------- #
# THE TRAP. READ THIS BEFORE CHANGING THE DEFAULT BELOW.
#
# There are TWO files named multi_positive_labels_v1.jsonl. IDENTICAL filename,
# IDENTICAL schema, both ~147 MB, both parse cleanly:
#
#   /home/alex/dh2_takehome/data/multi_positive_labels_v1.jsonl          <- THIS ONE
#       the post-re-merge store. 853 judged queries -> 123 intersection.
#   /home/alex/discovery_hub/data/pipeline2/train_labels/
#       multi_positive_labels_v1.jsonl                                   <- STALE
#       pre-re-merge. 213 judged queries -> 31 intersection.
#
# The stale one is the path dh2/config2.py actually points at, so reaching for
# "the" label store by muscle memory gets the wrong file. It FAILS SILENTLY: no
# error, no exception, no empty result -- just a 4x smaller slice that matches no
# number in config.py, and a harness that reports a confident rate over 31 queries
# while its docstring claims 123. A wrong number that looks right is worse than a
# crash, so the default is PINNED to the correct absolute path rather than derived
# from any config that could drift onto the stale copy.
# --------------------------------------------------------------------------- #
DEFAULT_LABEL_STORE = _env_path(
    "DH_LABEL_STORE",
    Path("/home/alex/dh2_takehome/data/multi_positive_labels_v1.jsonl"))

# The 440 held-out, LLM-adjudicated utility queries, graded 0-3.
DEFAULT_UTILITY_QRELS = _env_path(
    "DH_UTILITY_QRELS",
    Path("/home/alex/dh2_takehome/data/qrels/qrels_scout_utility_v1.jsonl"))

# The 1.5 GB / 603,369-doc normalized corpus.
#
# NOT derived from config.NORM_DIR on purpose. config.NORM_DIR defaults to
# ./data/normalized, which DOES NOT EXIST in this repo -- it only resolves to the
# real corpus once demo/env.sh has exported DH_DATA_ROOT=/home/alex/discovery_hub/
# data. Deriving from it would make this module's behaviour depend on whether the
# caller sourced env.sh, which is precisely the kind of invisible coupling that
# produces a "why is my evidence empty" afternoon. Pinned absolute; override with
# DH_STRICT_DOCS.
DEFAULT_DOCS = _env_path(
    "DH_STRICT_DOCS",
    Path("/home/alex/discovery_hub/data/normalized/docs.jsonl"))

# Utility-qrels grade semantics. relevant = grade >= 2 -- the same cut every
# retrieval number in config.py was computed under. Changing it silently
# re-defines "relevant" and de-comparables this slice from the whole repo.
RELEVANT_GRADE = 2
ACTIONABLE_GRADE = 3

# doc_id extracted from the RAW line before json.loads. The corpus is 603,369
# records / 1.5 GB and json.loads is ~99.7% of the cost of a naive scan; only
# 1,772 lines are ever wanted, so parsing the other 601,597 is pure waste.
# Anchored on the "doc_id" KEY rather than a bare substring search for the id
# itself: "uspto:US9717759B2" also appears inside source_url, and matching that
# would parse (and key) records on a URL coincidence.
_DOC_ID_RE = re.compile(r'"doc_id"\s*:\s*"([^"]+)"')


# --------------------------------------------------------------------------- #
# The query type
# --------------------------------------------------------------------------- #
@dataclass
class StrictQuery(EvalQuery):
    """
    An EvalQuery that remembers WHICH of its positives are directly actionable.

    Subclasses discovery_hub.evaluate.EvalQuery rather than parallelling it, so
    everything that already eats an EvalQuery (evaluate.write_qrels, 10's
    run_eval) eats this unchanged. Dataclass inheritance is checked and CLEAN
    here: EvalQuery's three fields all lack defaults, so appending one more
    field without a default is legal -- field order is
    (query_id, query, relevant_doc_ids, grade3_doc_ids). If a default is ever
    added to EvalQuery, this class breaks LOUDLY at import with TypeError
    ("non-default argument follows default argument"), which is the failure mode
    to want -- composition would have hidden the drift.

    grade3_doc_ids is a SUBSET of relevant_doc_ids (grade 3 is >= the grade 2 cut),
    not a disjoint bucket. It exists so as_candidates() can put the strongest
    evidence first without re-reading the qrels.

    NOTE: evaluate.load_qrels() returns plain EvalQuery, so a StrictQuery that is
    round-tripped through write_qrels/load_qrels comes back WITHOUT grade3_doc_ids.
    Re-derive from adjudicated_slice(); do not reconstruct it by hand.
    """

    grade3_doc_ids: list[str]


# --------------------------------------------------------------------------- #
# The judged store -> the 853
# --------------------------------------------------------------------------- #
def judged_query_ids(label_store=None) -> set[str]:
    """
    Query ids carrying at least one INDEPENDENT LLM adjudication.

    A row counts when llm_grade is not None AND masked is falsey:
      * llm_grade is None on rows that were never sent to the adjudicator (they
        carry a teacher-derived `grade` instead). Reading `grade` here instead of
        `llm_grade` would silently widen the set far past 853 -- `grade` is
        populated on essentially every row, including easy_negatives.
      * masked=True rows were withheld. Counting them re-admits exactly what the
        mask exists to exclude.

    Row-level filter, query-level result: one adjudicated row is enough to make
    the query judged.

    Returns exactly 853 on DEFAULT_LABEL_STORE (measured). If you get 213, you are
    reading the STALE copy -- see the trap comment on DEFAULT_LABEL_STORE.
    """
    path = Path(label_store) if label_store is not None else DEFAULT_LABEL_STORE
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("llm_grade") is None or row.get("masked"):
                continue
            out.add(str(row["query_id"]))
    return out


def _read_utility(utility_qrels=None) -> dict[str, dict]:
    """The utility qrels keyed by query_id. Each row: {query_id, query, grades{}}."""
    path = Path(utility_qrels) if utility_qrels is not None else DEFAULT_UTILITY_QRELS
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["query_id"])] = row
    return out


def _grade(value) -> int:
    """Grades are ints in the real file; coerce defensively, unreadable -> 0 (drop)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# The 123
# --------------------------------------------------------------------------- #
def adjudicated_slice(label_store=None, utility_qrels=None) -> list[StrictQuery]:
    """
    The utility-qrels queries that are ALSO independently LLM-adjudicated.

    = judged_query_ids() INTERSECT utility qrels = exactly 123 (measured). This is
    the slice config.py quotes every retrieval number against.

    Deterministic: iterated in sorted query_id order, and every doc_id list is
    sorted. No RNG anywhere in this module.
    """
    judged = judged_query_ids(label_store)
    rows = _read_utility(utility_qrels)

    out: list[StrictQuery] = []
    for qid in sorted(rows):
        if qid not in judged:
            continue
        grades = rows[qid].get("grades") or {}
        out.append(StrictQuery(
            query_id=qid,
            # The query TEXT is taken from the utility qrels, not the label store.
            # Both carry it and they agree, but the qrels row is the one that
            # defines this slice -- one source per field, so they cannot drift.
            query=rows[qid]["query"],
            relevant_doc_ids=sorted(d for d, g in grades.items()
                                    if _grade(g) >= RELEVANT_GRADE),
            grade3_doc_ids=sorted(d for d, g in grades.items()
                                  if _grade(g) >= ACTIONABLE_GRADE),
        ))
    return out


def select(n: int = 50, min_rel: int = RELEVANT_GRADE, label_store=None,
           utility_qrels=None) -> list[StrictQuery]:
    """
    The n best-covered queries of the 123 -- THE 50-query set the harnesses run on.

    min_rel is a COUNT of relevant docs (grade >= 2), NOT a grade. min_rel=2 means
    "at least two relevant documents", and it happens to equal RELEVANT_GRADE by
    coincidence of value, not of meaning.

    Sorted by (-len(relevant_doc_ids), query_id): best-covered first, ties broken
    lexicographically. Fully deterministic, NO RNG -- two calls return the same 50
    in the same order, which is what makes a re-run comparable to a prior run.

    Coverage is the selection criterion because a query with ONE relevant doc
    cannot distinguish "cited the evidence" from "cited the only thing on offer".
    It is NOT a claim that these 50 are representative -- they are the EASIEST 50
    by evidence density, and they skew away from pharma (see the module docstring's
    scope caveat). Any rate measured here is an upper-ish bound on a rate over the
    whole 123, and must be reported as "the 50 best-covered", never as "the corpus".

    Returns FEWER than n when fewer than n queries clear min_rel -- e.g. min_rel=3
    tops out at 45 (measured). It does NOT pad, and it does NOT fall back to a
    looser filter: silently returning 45 rows to a caller that asked for 50 and
    reports "n=50" is how a denominator goes wrong. CHECK len() ON THE RESULT.
    """
    pool = [q for q in adjudicated_slice(label_store, utility_qrels)
            if len(q.relevant_doc_ids) >= min_rel]
    pool.sort(key=lambda q: (-len(q.relevant_doc_ids), q.query_id))
    return pool[:n]


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def load_evidence(queries, docs_path=None) -> dict[str, dict]:
    """
    Resolve every relevant doc_id in `queries` to its text. ONE streaming pass.

    doc_id -> {"title", "abstract", "source", "source_url", "organizations"}

    THE FIELD IS "abstract". THERE IS NO "text" FIELD IN THIS CORPUS. A loader
    written as d.get("text") returns "" for all 603,369 records, and NOTHING
    ERRORS: synthesis drops every candidate for having no abstract and abstains,
    the arm scores 100% refusal, and the harness reports a beautiful grounding
    number produced by an evidence set that was empty the whole time. The corpus
    keys are: abstract, cpc_codes, doc_id, embedding_text, experts, extra,
    facilities, inventors, keywords, organizations, retrieved_date, source,
    source_url, title (checked against the real file).

    organizations is carried even though NO agent currently reads it -- 07's
    candidates have it, and as_candidates() is specified to emit it. Hardcoding []
    would be a quiet lie about a corpus field that is populated (e.g.
    uspto:US6803192B1 -> ['THE J DAVID GLADSTONE INST', 'UNIV CALIFORNIA']).

    Streams. It NEVER builds a dict of the corpus: 1.5 GB of records in RAM on a
    30 GB box that already OOM-killed a probe is not a tradeoff worth making for a
    1,772-record lookup. 07's Retriever does exactly that (self.docs = {d.doc_id:
    d for d in read_docs(...)}) and it is a large part of why a real Retriever is
    ~20.2 GB.

    Measured on select(): 1,772 distinct doc_ids resolved 1,772/1,772, 0 with an
    empty abstract, 5.3 s, peak RSS 32 MB -- i.e. ~600x cheaper in RAM than the
    ~20.2 GB Retriever this module exists to avoid constructing.
    """
    want: set[str] = set()
    for q in queries:
        want.update(q.relevant_doc_ids)
    if not want:
        return {}

    path = Path(docs_path) if docs_path is not None else DEFAULT_DOCS
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            # Cheap pre-filter: pull the doc_id out of the raw text and test set
            # membership BEFORE paying for json.loads. Short-circuits ~601,597 of
            # 603,369 lines.
            m = _DOC_ID_RE.search(line)
            if m is None or m.group(1) not in want:
                continue
            rec = json.loads(line)
            out[rec["doc_id"]] = {
                "title": rec.get("title", "") or "",
                "abstract": rec.get("abstract", "") or "",
                "source": rec.get("source", "") or "",
                "source_url": rec.get("source_url", "") or "",
                "organizations": rec.get("organizations") or [],
            }
            if len(out) == len(want):
                break  # every wanted doc found; no reason to read the rest
    return out


# --------------------------------------------------------------------------- #
# Gold docs -> 07-shaped candidates
#
# THE MISSING KEY IS THE POINT: no rerank_score.
#
# agents/synthesis.py line ~184 reads `reranked = "rerank_score" in kept[0]` --
# kept[0] ALONE, not any(). Gold candidates carry no cross-encoder score because NO
# CROSS-ENCODER RAN, so synthesis takes its mock branch and the 0.35 confidence
# gate is SKIPPED: payload["confidence_gate"] reads "NOT APPLIED: no reranker ran
# ...". That is a REAL LIMITATION of every harness built on this module and it must
# be stated in their reports, not discovered later.
#
# The fix is NOT to invent a rerank_score. A fabricated score would make the gate
# fire and produce a gate pass/fail rate that measures the number this module made
# up -- a measurement of a fiction, reported as a measurement of the system. If the
# gate needs measuring, run the real reranker over these candidates and let it
# write the field.
#
# "score" is 0.0 for the same reason, and it is NOT the qrels grade:
#   * grade/3 would leak the ANSWER KEY into the confidence the harness reports.
#   * 1.0 would assert maximal confidence for something no model ever scored.
#   * 0.0 means what is true -- NO RETRIEVAL SCORE EXISTS. It reaches nothing that
#     gates (synthesis only compares to 0.35 on the reranked branch; the
#     orchestrator never compares confidences at all -- checked), so it lands in
#     payload confidence as 0.0 next to "NOT APPLIED", which is self-consistent:
#     no retrieval ran, so there is no retrieval confidence.
# Likewise NO text_score/keyword_score/graph_score: agents/retrieval.py derives
# signals_fired from which of those are non-zero, so their absence correctly
# reports that NO retrieval arm fired.
# --------------------------------------------------------------------------- #
GOLD_SCORE = 0.0


def as_candidates(query: StrictQuery, evidence: dict[str, dict],
                  limit: int | None = None) -> list[dict]:
    """
    The gold docs for `query`, shaped exactly like 07's candidates.

    Keys: doc_id, title, abstract, source, source_url, score, organizations --
    the subset of 07's shape that agents/retrieval.py and agents/synthesis.py
    actually read, so both consume these unchanged. See the banner above for why
    rerank_score is absent and why score is 0.0.

    Order: grade-3 (directly_actionable) first, then grade-2, then by doc_id
    within each band. Deterministic. Strongest evidence first matters because
    synthesis truncates to max_claims=5 -- with 115 gold docs on q-0000038, the
    order decides what the model ever sees.

    Silently skips doc_ids absent from `evidence` (a doc the corpus does not
    carry grounds nothing). On the real slice nothing is skipped: 1,772/1,772
    resolve. If a caller needs to know about a gap, diff the doc_ids -- do not
    read it out of this list's length.
    """
    actionable = set(query.grade3_doc_ids)
    ordered = sorted(query.relevant_doc_ids, key=lambda d: (d not in actionable, d))

    out: list[dict] = []
    for doc_id in ordered:
        rec = evidence.get(doc_id)
        if rec is None:
            continue
        out.append({
            "doc_id": doc_id,
            "title": rec.get("title", ""),
            "abstract": rec.get("abstract", ""),
            "source": rec.get("source", ""),
            "source_url": rec.get("source_url", ""),
            "score": GOLD_SCORE,
            "organizations": rec.get("organizations") or [],
        })
        if limit is not None and len(out) >= limit:
            break
    return out
