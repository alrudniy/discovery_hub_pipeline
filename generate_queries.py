#!/usr/bin/env python3
"""
generate_queries.py -- synthetic query generation for embedder fine-tuning.

Drop-in for the `discovery_finetune` package. Manufactures (query, positive-doc)
training pairs with NO labeled data, using the InPars / Promptagator recipe with the
register-crossing twist that is the whole point of Discovery Hub:

    documents are written in legal/technical PATENT register;
    queries must be written in the clinical/business register a pharma scout uses.

For each normalized DiscoveryDoc we ask a generator LLM to read the technically-worded
invention and emit several queries in the *opposite* register -- the way a business-
development or clinical researcher would phrase the need -- deliberately NOT echoing the
document's jargon. Those (query -> doc) pairs are the positives `build_dataset.py` mines
hard negatives against and `train.py` optimizes with MultipleNegativesRankingLoss.

This version adds four things for a real (slow, rate-limited, paid-or-free) run:
  * RESUMABLE     -- every processed doc_id is recorded in a .done sidecar; re-running
                     skips finished docs (no repeat API calls). Hard API failures are
                     NOT marked done, so they retry next run.
  * TRAIN/EVAL SPLIT -- a deterministic per-doc_id hold-out (--eval-frac) writes eval
                     queries to a SEPARATE file in qrels format ({query_id, query,
                     relevant_doc_ids}) so 10_eval_retrieval.py can use it via --qrels.
                     Keeps eval queries out of the training pairs (no leakage).
  * SOURCE SCOPING -- --sources uspto,sbir generates only over the register-gap-heavy
                     sources (skip ClinicalTrials, which is already near scout register).
  * PROGRESS      -- periodic counter so a long run is observable.

Generator endpoint is any OpenAI-compatible API, configured with DEDICATED env vars so
the offline query-gen model stays independent of the online stage-08 explanation model
(DH_LLM_*). Defaults target Z.ai; use the FREE glm-4.7-flash first:

    export DH_QUERYGEN_BASE_URL="https://api.z.ai/api/paas/v4"
    export DH_QUERYGEN_API_KEY="sk-..."          # provider key (never commit)
    export DH_QUERYGEN_MODEL="glm-4.7-flash"     # free; step up to glm-4.7 / glm-5.2 if weak

Usage:
    # scoped, resumable real run (patents + SBIR, 10% held out for eval):
    python generate_queries.py --docs data/normalized/docs.jsonl \
        --out data/finetune/synthetic_queries.jsonl \
        --sources uspto,sbir --n-queries 3 --eval-frac 0.1 --max-docs 25000
    # re-run the SAME command to resume after an interruption.
    python generate_queries.py --mock --docs sample.jsonl --out out.jsonl   # no API/key
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

# --- dedicated env vars: keep the offline query-gen model independent of DH_LLM_* ---
QUERYGEN_BASE_URL = os.environ.get("DH_QUERYGEN_BASE_URL", "https://api.z.ai/api/paas/v4")
QUERYGEN_API_KEY = os.environ.get("DH_QUERYGEN_API_KEY", "")
QUERYGEN_MODEL = os.environ.get("DH_QUERYGEN_MODEL", "glm-4.7-flash")

MIN_CHARS_DEFAULT = 200          # noise filter: skip near-empty / boilerplate docs
MIN_Q_WORDS, MAX_Q_WORDS = 4, 40

# Few-shot exemplars that TEACH THE REGISTER CROSS (technical doc -> lay scout query).
FEWSHOT = [
    {
        "doc": ("Title: Bicyclic heteroaryl compounds as inhibitors of Bruton's tyrosine "
                "kinase. Abstract: Disclosed are substituted pyrazolo[3,4-d]pyrimidine "
                "derivatives that covalently bind Cys481 of BTK, pharmaceutical "
                "compositions thereof, and methods of treating B-cell proliferative "
                "disorders."),
        "queries": [
            "covalent BTK inhibitor for B-cell lymphoma we could in-license",
            "small molecule targeting Bruton's tyrosine kinase for autoimmune indications",
            "oral therapy for relapsed chronic lymphocytic leukemia",
        ],
    },
    {
        "doc": ("Title: Lipid nanoparticle formulations for delivery of messenger RNA. "
                "Abstract: Ionizable cationic lipids and processes for encapsulating mRNA "
                "payloads to enhance endosomal escape and in vivo expression in hepatic "
                "tissue."),
        "queries": [
            "mRNA delivery platform for liver-targeted gene therapy",
            "ionizable lipid nanoparticle technology available for partnership",
            "non-viral vector to improve in vivo expression of RNA therapeutics",
        ],
    },
]


def _client():
    if not QUERYGEN_API_KEY:
        sys.exit("DH_QUERYGEN_API_KEY is not set (needed for real mode; use --mock to "
                 "run without an API).")
    try:
        from openai import OpenAI  # OpenAI-compatible (Z.ai / vLLM / others)
    except ImportError:
        sys.exit("the 'openai' package is required for real mode: pip install openai")
    return OpenAI(base_url=QUERYGEN_BASE_URL, api_key=QUERYGEN_API_KEY)


def _build_prompt(doc_text: str, n: int) -> list[dict]:
    sys_msg = (
        "You generate search queries for a pharmaceutical technology-scouting engine. "
        "You are shown an invention written in dense legal/technical PATENT language. "
        "Produce queries in the DIFFERENT register a pharma business-development or "
        "clinical researcher actually uses: plain clinical/commercial language about the "
        "therapeutic goal, target, indication, modality, or licensing need. Crucially, do "
        "NOT copy the document's technical jargon, chemical names, or rare phrasing -- "
        "paraphrase the intent the way a person searching would phrase it. Do NOT reference "
        "specific compound codes, drug brand names, molecule identifiers, or trial/patent "
        "numbers -- a scout looking for technology to license would not already know them; "
        "describe the target, mechanism, modality, and indication generically. Vary the "
        "angle across queries (mechanism in lay terms, disease/indication, modality, "
        "business/licensing framing). "
        f"Return ONLY a JSON array of exactly {n} short query strings, nothing else."
    )
    msgs = [{"role": "system", "content": sys_msg}]
    for ex in FEWSHOT:                                  # few-shot demonstrations
        msgs.append({"role": "user", "content": ex["doc"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["queries"][:n])})
    msgs.append({"role": "user", "content": doc_text})
    return msgs


def _parse_array(text: str) -> list[str]:
    """Tolerate ```json fences / stray prose around the JSON array."""
    m = re.search(r"\[.*\]", text.strip(), re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(q) for q in arr if isinstance(q, str)]


def _gen_real(client, doc_text: str, n: int, temperature: float):
    """Returns a list on a successful API call (possibly empty after parsing), or
    None on hard failure (all retries exhausted) so the caller can retry it later."""
    for attempt in range(3):                            # retry network / rate / parse
        try:
            resp = client.chat.completions.create(
                model=QUERYGEN_MODEL, temperature=temperature,
                messages=_build_prompt(doc_text, n))
            return _parse_array(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                print(f"  ! generation failed ({e}); will retry this doc next run",
                      file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# ---- deterministic mock generator: derive lay queries without any API ----
_JARGON = re.compile(r"(derivativ|substitut|compositio|heteroaryl|bicyclic|pyrazolo|"
                     r"pyrimidin|ionizable|encapsulat|endosomal|method of|disclosed|"
                     r"comprising|wherein|embodiment)", re.I)
_TEMPLATES = [
    "therapy targeting {kw} for {ind}",
    "{mod} for {ind} available for licensing",
    "partner with technology for {kw}",
    "treatment approach involving {kw}",
    "{kw} candidate for clinical development",
]
_INDICATIONS = ["oncology", "autoimmune disease", "rare disease", "metabolic disorders",
                "neurology", "inflammatory conditions"]
_MODALITIES = ["small molecule", "biologic", "delivery platform", "antibody",
               "gene therapy", "cell therapy"]


def _keywords(doc) -> list[str]:
    kws = list(doc.get("keywords") or [])
    if not kws:                                         # fall back to non-jargon title words
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", doc.get("title", ""))
        kws = [w.lower() for w in words if not _JARGON.search(w)]
    return kws or ["novel therapeutic"]


def _gen_mock(doc, n: int) -> list[str]:
    rng = random.Random(int(hashlib.sha1(doc.get("doc_id", "").encode()).hexdigest(), 16))
    kws, out = _keywords(doc), []
    for _ in range(n * 3):
        q = rng.choice(_TEMPLATES).format(
            kw=rng.choice(kws), ind=rng.choice(_INDICATIONS), mod=rng.choice(_MODALITIES))
        if q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


# ---- quality filters shared by both modes ----
def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().strip('"').strip()


def _ok(q: str, title: str) -> bool:
    w = q.split()
    if not (MIN_Q_WORDS <= len(w) <= MAX_Q_WORDS):
        return False
    # reject near-verbatim copies of the title (enforce the register cross)
    tset, qset = set(title.lower().split()), set(q.lower().split())
    if tset and len(tset & qset) / max(1, len(qset)) > 0.8:
        return False
    return True


def _filter(qs, title):
    seen, out = set(), []
    for q in qs:
        q = _norm(q)
        if q and q.lower() not in seen and _ok(q, title):
            seen.add(q.lower())
            out.append(q)
    return out


# ---- resume + split helpers ----
def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text().splitlines() if l.strip()}


def _is_eval(doc_id: str, eval_frac: float, seed: int) -> bool:
    """Deterministic per-doc hold-out: stable across runs and independent of order."""
    if eval_frac <= 0:
        return False
    h = int(hashlib.sha1(f"{seed}:{doc_id}".encode()).hexdigest(), 16) % 10000
    return h < int(eval_frac * 10000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default="data/normalized/docs.jsonl",
                    help="input normalized DiscoveryDoc JSONL")
    ap.add_argument("--out", default="data/finetune/synthetic_queries.jsonl",
                    help="training pairs output (build_dataset.py consumes this)")
    ap.add_argument("--eval-out", default=None,
                    help="held-out eval queries in qrels format "
                         "(default: <out>.eval.jsonl)")
    ap.add_argument("--eval-frac", type=float, default=0.1,
                    help="fraction of docs held out for eval-query generation (0 = none)")
    ap.add_argument("--sources", default="",
                    help="comma-separated source filter, e.g. uspto,sbir (default: all)")
    ap.add_argument("--n-queries", type=int, default=3, help="queries per document")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT,
                    help="noise filter: skip docs with less text than this")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="0 = all; else cap TOTAL processed docs (incl. already-done)")
    ap.add_argument("--temperature", type=float, default=0.7, help="query diversity")
    ap.add_argument("--seed", type=int, default=20240611)
    ap.add_argument("--progress-every", type=int, default=500,
                    help="log progress every N newly-processed docs")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel in-flight API requests (keep <= the model's "
                         "concurrency limit; GLM-4.5/5.2 allow 10, so 8 is safe)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore existing output/.done and start fresh")
    ap.add_argument("--mock", action="store_true", help="no API/key; deterministic")
    args = ap.parse_args()

    random.seed(args.seed)
    docs_path, out_path = Path(args.docs), Path(args.out)
    eval_path = Path(args.eval_out) if args.eval_out else \
        out_path.with_suffix(out_path.suffix + ".eval.jsonl")
    done_path = out_path.with_suffix(out_path.suffix + ".done")
    if not docs_path.exists():
        sys.exit(f"input not found: {docs_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.restart:
        for p in (out_path, eval_path, done_path):
            if p.exists():
                p.unlink()

    sources_set = {s.strip() for s in args.sources.split(",") if s.strip()}
    done = _load_done(done_path)
    already = len(done)
    resuming = already > 0            # trust the .done ledger; use --restart for a clean slate
    fmode = "a" if resuming else "w"

    client = None if args.mock else _client()
    generator = "mock" if args.mock else QUERYGEN_MODEL
    today = date.today().isoformat()

    print(f"generate_queries: docs={docs_path} out={out_path}")
    print(f"  sources={sorted(sources_set) or 'all'} n_queries={args.n_queries} "
          f"eval_frac={args.eval_frac} max_docs={args.max_docs or 'all'} "
          f"generator={generator}{'' if args.mock else '  @ ' + QUERYGEN_BASE_URL}")
    if resuming:
        print(f"  RESUMING: {already} doc(s) already done -> skipping them, appending.")

    # ---- build the pending work list (fast: filters only, no API calls) ----
    def _doc_text(doc):
        return f"Title: {doc.get('title', '')}. Abstract: {doc.get('abstract', '')}".strip()

    pending, n_seen = [], 0
    with docs_path.open() as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            n_seen += 1
            did = doc.get("doc_id", "")
            if sources_set and doc.get("source") not in sources_set:
                continue
            text = (doc.get("embedding_text")
                    or f"{doc.get('title', '')} {doc.get('abstract', '')}").strip()
            if len(text) < args.min_chars:                  # noise filter
                continue
            if did in done:                                 # already done on a prior run
                continue
            pending.append(doc)
            if args.max_docs and (already + len(pending)) >= args.max_docs:
                break
    print(f"  scanned {n_seen} docs; {len(pending)} to process "
          f"(concurrency={args.concurrency})")

    def _generate(doc):
        if args.mock:
            return _gen_mock(doc, args.n_queries)
        return _gen_real(client, _doc_text(doc), args.n_queries, args.temperature)

    # ---- generate concurrently; serialize file writes + counters under a lock ----
    lock = threading.Lock()
    ctr = {"new": 0, "train": 0, "eval": 0, "failed": 0}
    t0 = time.time()
    with out_path.open(fmode) as f_train, eval_path.open(fmode) as f_eval, \
            done_path.open("a") as f_done, \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(_generate, d): d for d in pending}
        for fut in as_completed(futs):
            doc = futs[fut]
            did = doc.get("doc_id", "")
            raw = fut.result()                  # list on success, None on hard API failure
            with lock:                          # only the fast write path is serialized
                if raw is None:
                    ctr["failed"] += 1
                    continue
                queries = _filter(raw, doc.get("title", ""))[:args.n_queries]
                if queries:
                    if _is_eval(did, args.eval_frac, args.seed):
                        for i, q in enumerate(queries):
                            f_eval.write(json.dumps({
                                "query_id": f"{did}#{i}", "query": q,
                                "relevant_doc_ids": [did], "source": doc.get("source"),
                                "generator": generator, "generated_date": today}) + "\n")
                            ctr["eval"] += 1
                    else:
                        for i, q in enumerate(queries):
                            f_train.write(json.dumps({
                                "query": q, "doc_id": did, "source": doc.get("source"),
                                "source_url": doc.get("source_url", ""),
                                "retrieved_date": doc.get("retrieved_date", ""),
                                "generator": generator, "generated_date": today,
                                "query_index": i}) + "\n")
                            ctr["train"] += 1
                # mark done AFTER writing (successful attempt, even if 0 survived)
                f_train.flush(); f_eval.flush()
                f_done.write(did + "\n"); f_done.flush()
                done.add(did)
                ctr["new"] += 1
                if ctr["new"] % args.progress_every == 0:
                    rate = ctr["new"] / max(time.time() - t0, 1e-6)
                    print(f"  processed {ctr['new']} new ({already + ctr['new']} total) | "
                          f"train_pairs={ctr['train']} eval_pairs={ctr['eval']} "
                          f"failed={ctr['failed']} | {rate:.1f} docs/s", flush=True)

    print(f"\ndone. new docs={ctr['new']} (total {already + ctr['new']}) | "
          f"train pairs={ctr['train']} -> {out_path}")
    print(f"      eval pairs={ctr['eval']} -> {eval_path}")
    if ctr["failed"]:
        print(f"      {ctr['failed']} doc(s) failed the API and were NOT marked done -- "
              f"re-run the same command to retry just those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
