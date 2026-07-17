#!/usr/bin/env python3
"""
08_multiagent_rag.py  --  LAYER 3 (Explanation & Workflow).

A strict-RAG, multi-agent pipeline that turns retrieved candidates into a cited,
provenance-stamped, confidence-scored recommendation. Implemented framework-free
as a state dict flowing through agent functions (the same shape maps 1:1 onto
LangGraph nodes for the real build).

  AGENTS
    1. retrieval        -> pull candidates (delegates to 07's Retriever)
    2. expertise_gap    -> annotate which experts/orgs/facilities back each tech;
                           flag candidates lacking provenance
    3. reranking        -> final policy-aware ordering (completeness + score)
    4. policy_safety    -> strict-RAG gates: drop evidence-free candidates, refuse
                           if top confidence < threshold, require a citation per claim
    5. explanation      -> the LLM (mock template or vLLM/Gemini) writes the prose,
                           citing ONLY retrieved sources

  TARGET: Drew (always-on). With a self-hosted LLM you control batching/seeds and
          can approach reproducibility; an API model trades that control for power.

  HUMAN-IN-THE-LOOP: output is a shortlist + explanation. People decide.

Usage:
  python 08_multiagent_rag.py --mock --query "kinase inhibitor for oncology"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism


def _load_retriever_module():
    """Import 07_retrieve_rank.py by path (module name starts with a digit)."""
    path = Path(__file__).with_name("07_retrieve_rank.py")
    spec = importlib.util.spec_from_file_location("retrieve_rank", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _confidence(score: float) -> float:
    """
    Confidence = the cross-encoder reranker's relevance score for the candidate.

    The reranker is the pipeline's most accurate relevance judge (it's the final
    ranking stage), so its score is the right confidence signal. Earlier versions
    read the dense cosine (``text_score``) instead, but that badly under-scores
    candidates surfaced by the keyword/graph arms: e.g. a clinical trial that BM25
    correctly retrieves for a clinical-language query can sit far from that query in
    the embedding space (dense cosine ~0.04) while the cross-encoder rightly scores
    it ~0.9. Reading dense cosine there caused the policy gate to refuse genuine
    matches. BGE reranker scores are already ~[0,1]; we sigmoid-guard only for any
    out-of-range value. Real deployments should still calibrate against labeled
    relevance (e.g. Platt scaling on the eval set).
    """
    if 0.0 <= score <= 1.0:
        return round(score, 4)
    import math
    return round(1.0 / (1.0 + math.exp(-score)), 4)   # squash raw logits to [0,1]


# --------------------------------------------------------------------------- #
# Agents: each takes and returns the shared state dict.
# --------------------------------------------------------------------------- #
def agent_retrieval(state: dict) -> dict:
    cands = state["_retriever"].retrieve(state["query"], rerank_k=state["k"])
    state["candidates"] = cands
    return state


def agent_expertise_gap(state: dict) -> dict:
    for c in state["candidates"]:
        backers = list(c.get("organizations", []))
        c["expertise"] = backers
        c["expertise_gap"] = (len(backers) == 0)  # no org/expert linked => gap
    state["num_gaps"] = sum(c["expertise_gap"] for c in state["candidates"])
    return state


def agent_reranking(state: dict) -> dict:
    # Policy-aware final order: prefer complete provenance, then score.
    state["candidates"].sort(
        key=lambda c: (not c["expertise_gap"], c["score"]), reverse=True)
    return state


def agent_policy_safety(state: dict) -> dict:
    # Strict RAG: keep only candidates that actually carry evidence text.
    kept = [c for c in state["candidates"]
            if (c.get("abstract") or "").strip() and c.get("source_url")]
    state["candidates"] = kept
    # Confidence reads the cross-encoder rerank score (the calibrated relevance
    # signal), not the dense cosine — see _confidence. Use the top candidate's score.
    top = max((c.get("rerank_score", 0.0) for c in kept), default=0.0)
    state["overall_confidence"] = _confidence(top)
    # Refuse to assert a recommendation we cannot ground / are not confident in.
    if not kept:
        state["refusal"] = "No evidence-backed candidates found; nothing to recommend."
    elif state["overall_confidence"] < config.RETRIEVAL.min_confidence:
        state["refusal"] = (f"Top confidence {state['overall_confidence']:.2f} below "
                            f"threshold {config.RETRIEVAL.min_confidence}; flagged for "
                            f"human review rather than asserted.")
    else:
        state["refusal"] = None
    return state


def _explain_mock(query: str, c: dict) -> str:
    """Deterministic stand-in for the LLM. Cites the real source; no free text
    beyond the retrieved evidence (strict RAG)."""
    snippet = (c.get("abstract") or "")[:160].rstrip()
    org = (c.get("organizations") or ["an unnamed organization"])[0]
    return (f"Relevant to '{query}': {c.get('title','(untitled)')} from {org}. "
            f"Supporting evidence: \"{snippet}...\" [source: {c['source_url']}]")


def agent_explanation(state: dict, explain_fn) -> dict:
    recs = []
    for c in state["candidates"]:
        recs.append({
            "doc_id": c["doc_id"],
            "title": c.get("title", ""),
            "why": explain_fn(state["query"], c),
            "citation": c["source_url"],          # every claim carries a citation
            "confidence": _confidence(c.get("rerank_score", 0.0)),
            "provenance": {"source": c.get("source"),
                           "retrieved_via": "DiscoveryHub/retrieve_rank"},
            "expertise_gap": c["expertise_gap"],
        })
    state["recommendations"] = recs
    return state


def run_pipeline(query: str, mock: bool = True, k: int | None = None,
                 retriever=None, explain_fn=None) -> dict:
    """Execute the agent graph and return the structured, cited result."""
    k = k or config.RETRIEVAL.top_k_rerank
    if retriever is None:
        rr = _load_retriever_module()
        retriever = rr.Retriever(mock=mock)
    explain_fn = explain_fn or (_explain_mock if mock else _explain_real_factory())

    state = {"query": query, "k": k, "_retriever": retriever}
    for agent in (agent_retrieval, agent_expertise_gap,
                  agent_reranking, agent_policy_safety):
        state = agent(state)
    if state["refusal"] is None:
        state = agent_explanation(state, explain_fn)
    else:
        state["recommendations"] = []

    state.pop("_retriever", None)
    state.pop("candidates", None)
    return {
        "query": query,
        "overall_confidence": state["overall_confidence"],
        "refusal": state["refusal"],
        "num_expertise_gaps": state.get("num_gaps", 0),
        "recommendations": state["recommendations"],
        "human_in_the_loop": "Shortlist only. A person reviews and decides.",
    }


def _explain_real_factory():
    """Returns an explain_fn backed by the 1min.ai unified chat API, strict RAG.

    Config via env (no hardcoded secrets):
      DH_LLM_API_KEY   -- 1min.ai API key (required for real mode)
      DH_LLM_MODEL     -- model name, e.g. 'gpt-4o-mini' (default below)
      DH_LLM_BASE_URL  -- override endpoint (default https://api.1min.ai)
    The response text lives at aiRecord.aiRecordDetail.resultObject[0].
    """
    import os
    import requests

    api_key = os.environ.get("DH_LLM_API_KEY")
    if not api_key:
        raise RuntimeError("DH_LLM_API_KEY not set; needed for non-mock explanation. "
                           "Run with mock=True, or export DH_LLM_API_KEY.")
    model = os.environ.get("DH_LLM_MODEL", "gpt-4o-mini")
    base = os.environ.get("DH_LLM_BASE_URL", "https://api.1min.ai").rstrip("/")
    url = f"{base}/api/chat-with-ai"          # non-streaming
    headers = {"Content-Type": "application/json", "API-KEY": api_key}

    def explain(query: str, c: dict) -> str:
        prompt = (
            "You are a strict-RAG assistant for a pharmaceutical technology-scouting "
            "tool. Using ONLY the evidence below, write ONE concise sentence explaining "
            "why this record matches the research interest, then cite the source URL in "
            "square brackets. Do not add any facts not present in the evidence.\n\n"
            f"Research interest: {query}\n"
            f"Title: {c.get('title')}\n"
            f"Evidence: {(c.get('abstract') or '')[:1200]}\n"
            f"Source: {c['source_url']}")
        payload = {
            "type": "UNIFY_CHAT_WITH_AI",
            "model": model,
            "promptObject": {"prompt": prompt},
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # non-streaming success -> aiRecord.aiRecordDetail.resultObject: [str, ...]
            result = (data.get("aiRecord", {})
                          .get("aiRecordDetail", {})
                          .get("resultObject", []))
            text = (result[0] if isinstance(result, list) and result else "").strip()
            if not text:
                raise ValueError(f"empty resultObject in response: {data}")
            # strict-RAG guard: guarantee the citation is present even if the model omits it
            if c["source_url"] not in text:
                text = f"{text} [source: {c['source_url']}]"
            return text
        except Exception as e:
            # fail safe to the deterministic, cited template rather than dropping the item
            return _explain_mock(query, c) + f"  (LLM unavailable: {type(e).__name__})"
    return explain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=config.RETRIEVAL.top_k_rerank)
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    result = run_pipeline(args.query, mock=args.mock, k=args.k)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
