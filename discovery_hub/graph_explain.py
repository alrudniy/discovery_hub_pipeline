"""
Graph EXPLANATION -- the graph's job, after the measurements took retrieval away.

WHY THIS EXISTS, AND WHY IT IS NOT THE RETRIEVAL CHANNEL
The entity-linked graph retrieval channel is off (config.RETRIEVAL.use_graph=False)
on direct evidence: across the 62 firing queries of the judged slice it surfaced
ZERO relevant documents that dense's top-100 missed, and in its full 600,738-doc
ranking the median best rank of any relevant doc was 195,898. The root cause is
upstream of the R-GCN -- ~1 of 853 judged queries names a real organization, so
query->entity linking has nothing true to link and fires only on junk.

THAT ARGUMENT DOES NOT TOUCH THIS MODULE. Explanation never links the query. It
starts from doc_ids the retriever ALREADY returned and walks real edges outward.
So:
  * a 0% query link rate is irrelevant here -- no query is linked;
  * there is no pool, no RRF, no displacement: retrieval is not consulted or
    changed, and this cannot cost recall;
  * every fact is derived from edges that exist, and carries them as evidence.

WHAT IT ANSWERS
Given the results a scout is already looking at:
  "this patent shares an inventor with the one you found"
  "that inventor is affiliated with Yale"
  "BMS and MedImmune are the assignees active in this space"
This is the same structure chen_pdl1_story.svg draws, computed for any result set.

WHAT THE EVIDENCE GUARANTEE IS, AND WHAT IT IS NOT
Every Fact carries `evidence`: the exact (src, rel, dst) triples it was derived
from, and verify() re-checks each against the loaded edge set.

That buys AUDITABILITY, not truth. It proves the generator is FAITHFUL TO THE GRAPH
-- i.e. that this module has no bug that invents edges. It says nothing about
whether the edge itself is true, and this graph is known to carry garbage at scale:
112,237 of its 195,557 "organization" nodes were individuals mistyped by USPTO
assignee parsing, and inventor nodes are keyed on normalized name strings
("inventor:chen lieping"), so distinct people with the same name collapse into one
node -- CHEN SHUHUI has 83 distinct assignees across 237 patents, which is not a
person. A surviving bad edge passes verify() with a clean 0-not-in-graph and prints
with full confidence.

So the claim to make is: "every sentence traces to a named triple, so when it is
wrong you can point at the edge that made it wrong." That is a stronger claim than
"it cannot fabricate", because it survives contact with a domain expert in the
room. Do not pitch this module as incapable of error. It is capable of repeating
every error the graph contains, precisely and with citations.

`inventor_risk` on a crossover fact is the mitigation that follows from that: it
carries the inventor's distinct-assignee count so a reader can discount a fact
routed through a probable merged phantom instead of trusting it uniformly.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Relations we walk. Kept explicit: a new relation should be an intentional
# addition here, not something that silently starts appearing in explanations.
_INVENTED_BY = "invented_by"
_ASSIGNED_TO = "assigned_to"
_AFFILIATED_WITH = "affiliated_with"


@dataclass
class Fact:
    kind: str                      # same_family | shared_inventor_crossover |
                                   # assignee | affiliation | space
    text: str                      # the sentence a person reads
    evidence: list[tuple[str, str, str]] = field(default_factory=list)  # (src, rel, dst)
    # Distinct assignees of the inventor this fact routes through, when it routes
    # through one. Inventor nodes are name-slug keyed, so a common name fuses many
    # people into one node and the fusion RENDERS AS A CROSSOVER -- the false
    # positives land on exactly the fact type we want to headline. A real career
    # spans a few employers; CHEN SHUHUI's node spans 83. Carried, not hidden, so a
    # reader can discount rather than trust uniformly.
    inventor_risk: int | None = None
    # For pair facts: the two doc_ids the relationship holds BETWEEN, canonical
    # (sorted) order. Present so a caller can attach the fact to both results for
    # display without the generator emitting it twice.
    docs: list[str] | None = None

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "text": self.text,
             "evidence": [list(e) for e in self.evidence]}
        if self.inventor_risk is not None:
            d["inventor_risk"] = self.inventor_risk
        if self.docs is not None:
            d["docs"] = list(self.docs)
        return d


class GraphExplainer:
    """Walks the KG outward from retrieved documents. Read-only, no query linking.

    Loads nodes.jsonl + edges.jsonl but NOT rgcn_node_emb.npy: explanation is pure
    graph structure, so it costs nothing on the GPU and does not need the R-GCN.
    """

    def __init__(self, graph_dir: str | Path):
        gdir = Path(graph_dir)
        self.label: dict[str, str] = {}
        self.ntype: dict[str, str] = {}
        self.docid_to_node: dict[str, str] = {}
        for line in (gdir / "nodes.jsonl").open():
            if not line.strip():
                continue
            n = json.loads(line)
            nid = n["node_id"]
            self.label[nid] = n.get("label") or nid
            self.ntype[nid] = n.get("ntype")
            did = n.get("doc_id")
            if did and n.get("ntype") == "technology":
                self.docid_to_node[did] = nid

        self.adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._edges: set[tuple[str, str, str]] = set()
        for line in (gdir / "edges.jsonl").open():
            if not line.strip():
                continue
            e = json.loads(line)
            s, d, r = e["src"], e["dst"], e["rel"]
            self.adj[s].append((d, r))
            self.adj[d].append((s, r))
            self._edges.add((s, d, r))
        self._inv_assignees = self._score_inventor_risk()

    def _score_inventor_risk(self) -> dict[str, int]:
        """inventor -> number of DISTINCT organizations across their patents.

        The same tell that exposed the phantom orgs. A prolific researcher accrues a
        handful of employers over a career; a name-slug node that fuses every
        "Zhang Wei" in the corpus accrues dozens. Computed once at load: it is the
        only thing standing between a false merge and a headline crossover.
        """
        inv2pat: dict[str, set[str]] = defaultdict(set)
        pat2asg: dict[str, set[str]] = defaultdict(set)
        for src, lst in self.adj.items():
            if self.ntype.get(src) != "technology":
                continue
            for nbr, rel in lst:
                if rel == _INVENTED_BY and self.ntype.get(nbr) == "inventor":
                    inv2pat[nbr].add(src)
                elif rel == _ASSIGNED_TO and self.ntype.get(nbr) == "organization":
                    pat2asg[src].add(nbr)
        out: dict[str, int] = {}
        for inv, pats in inv2pat.items():
            orgs: set[str] = set()
            for p in pats:
                orgs |= pat2asg.get(p, set())
            out[inv] = len(orgs)
        return out

    # -- primitives ---------------------------------------------------------
    def _nbrs(self, nid: str, rel: str, want_type: str | None = None):
        out = []
        for nbr, r in self.adj.get(nid, ()):
            if r != rel:
                continue
            if want_type and self.ntype.get(nbr) != want_type:
                continue
            out.append(nbr)
        return sorted(set(out))

    def _edge(self, a: str, b: str, rel: str) -> tuple[str, str, str]:
        """Return the triple in the direction it actually exists."""
        return (a, b, rel) if (a, b, rel) in self._edges else (b, a, rel)

    def inventors(self, doc_id: str) -> list[str]:
        nid = self.docid_to_node.get(doc_id)
        return self._nbrs(nid, _INVENTED_BY, "inventor") if nid else []

    def assignees(self, doc_id: str) -> list[str]:
        nid = self.docid_to_node.get(doc_id)
        return self._nbrs(nid, _ASSIGNED_TO, "organization") if nid else []

    # -- the explanation ----------------------------------------------------
    def explain(self, doc_ids: list[str], *, max_per_kind: int = 3,
                max_pairs_per_inventor: int = 3) -> dict:
        """Explain a RESULT SET. Cross-result facts are the point: the scout wants
        to know how the things in front of them connect, which is exactly what a
        ranked list cannot show.

        Returns three sections, and the split is load-bearing:
          per_doc[d] -- properties OF one result (assignee, affiliation)
          pairs      -- relationships BETWEEN two results, each emitted ONCE
          space      -- aggregates across the whole set

        `pairs` used to live in per_doc, which double-counted every relationship:
        A->B and B->A both printed, so a histogram over 853 judged queries reported
        8,852 crossovers where only 4,026 exist. The tell was that almost every
        per-query count was EVEN. A relationship is not a property of one endpoint,
        so it does not belong under one endpoint.
        """
        known = [d for d in doc_ids if d in self.docid_to_node]
        inv_of = {d: self.inventors(d) for d in known}
        asg_of = {d: self.assignees(d) for d in known}

        # inventor -> which of THESE results they invented
        by_inventor: dict[str, list[str]] = defaultdict(list)
        for d, invs in inv_of.items():
            for i in invs:
                by_inventor[i].append(d)

        # ---- properties of a single result ----
        per_doc: dict[str, list[Fact]] = {d: [] for d in doc_ids}
        for d in known:
            facts = per_doc[d]
            for a in asg_of[d][:max_per_kind]:
                facts.append(Fact(
                    "assignee", f"Assigned to {self.label[a]}",
                    [self._edge(self.docid_to_node[d], a, _ASSIGNED_TO)]))
            for i in inv_of[d][:max_per_kind]:
                for org in self._nbrs(i, _AFFILIATED_WITH, "organization")[:2]:
                    facts.append(Fact(
                        "affiliation",
                        f"Inventor {self.label[i]} is affiliated with {self.label[org]}",
                        [self._edge(self.docid_to_node[d], i, _INVENTED_BY),
                         self._edge(i, org, _AFFILIATED_WITH)]))

        # ---- relationships between two results: ONCE, canonical order ----
        # Split on whether the other result is the SAME invention. Measured on a real
        # PD-L1 query, every shared-inventor pair in the top-8 was a same-titled
        # continuation: true, and worthless. The crossover (different invention, same
        # person) is the B7-H1 -> PD-L1 story. Different features; never conflate.
        pairs: list[Fact] = []
        truncated: list[dict] = []
        for i, docs in by_inventor.items():
            if len(docs) < 2:
                continue
            ds = sorted(set(docs))
            combos = [(a, b) for x, a in enumerate(ds) for b in ds[x + 1:]]
            # An inventor on a long analog series (apelin/APJ: one team, dozens of
            # near-identical compound patents) generates k*(k-1)/2 true-but-identical
            # pairs and buries everything else. Cap, and SAY what was dropped -- a
            # silent cap reads as "that's all there is".
            #
            # CLASSIFY BEFORE CAPPING, and give each kind its own budget. Capping the
            # raw combo list first was measured to destroy the thing it protects:
            # combos[:3] takes pairs in doc order, so an inventor whose first three
            # pairs happen to be same-title continuations keeps three worthless facts
            # and drops the crossover. Over the 853 judged queries that silently ate
            # ~1,055 crossovers and cost 6 queries their only one -- a cap that
            # preferentially discards the signal.
            cross_c = [(a, b) for a, b in combos if not self._same_invention(a, b)]
            fam_c = [(a, b) for a, b in combos if self._same_invention(a, b)]
            keep = (cross_c[:max_pairs_per_inventor] + fam_c[:max_pairs_per_inventor])
            for a, b in keep:
                same = self._same_invention(a, b)
                pairs.append(Fact(
                    "same_family" if same else "shared_inventor_crossover",
                    (f"{self.label[i]} invented both, and they are the same invention "
                     f"family: {self._title(a)}" if same else
                     f"{self.label[i]} invented BOTH {self._title(a)} AND a different "
                     f"invention: {self._title(b)}"),
                    [self._edge(self.docid_to_node[a], i, _INVENTED_BY),
                     self._edge(self.docid_to_node[b], i, _INVENTED_BY)],
                    inventor_risk=self._inv_assignees.get(i),
                    docs=[a, b]))
            if len(combos) > len(keep):
                truncated.append({
                    "inventor": self.label[i], "shown": len(keep),
                    "dropped": len(combos) - len(keep),
                    "dropped_crossover": max(0, len(cross_c) - max_pairs_per_inventor),
                    "dropped_family": max(0, len(fam_c) - max_pairs_per_inventor),
                    "results_linked": len(ds),
                    "inventor_risk": self._inv_assignees.get(i)})

        # 4. who is active across the whole result set
        space: list[Fact] = []
        org_hits = Counter()
        for d, orgs in asg_of.items():
            for a in orgs:
                org_hits[a] += 1
        for org, c in org_hits.most_common(5):
            if c < 2:
                continue
            docs = [d for d in known if org in asg_of[d]]
            space.append(Fact(
                "space",
                f"{self.label[org]} is an assignee on {c} of these {len(known)} results",
                [self._edge(self.docid_to_node[d], org, _ASSIGNED_TO) for d in docs]))

        inv_hits = Counter({i: len(v) for i, v in by_inventor.items() if len(v) > 1})
        for i, c in inv_hits.most_common(5):
            docs = by_inventor[i]
            space.append(Fact(
                "space",
                f"{self.label[i]} is an inventor on {c} of these {len(known)} results",
                [self._edge(self.docid_to_node[d], i, _INVENTED_BY) for d in docs]))

        return {"per_doc": {d: [f.to_dict() for f in fs] for d, fs in per_doc.items()},
                "pairs": [f.to_dict() for f in pairs],
                "space": [f.to_dict() for f in space],
                "truncated": truncated,
                "coverage": {"requested": len(doc_ids), "in_graph": len(known)}}

    @staticmethod
    def facts_for(explanation: dict, doc_id: str) -> list[dict]:
        """Everything a UI should show against ONE result: its own properties plus
        the relationships it participates in. The pair is stored once and read
        twice -- display symmetry without double counting."""
        out = list(explanation["per_doc"].get(doc_id, []))
        out += [p for p in explanation.get("pairs", []) if doc_id in (p.get("docs") or [])]
        return out

    def _same_invention(self, a: str, b: str) -> bool:
        """Two results are the same invention when their titles match after
        normalisation. Deliberately crude and deliberately conservative: it only
        ever DOWNGRADES a claim from 'crossover' to 'family', so a miss costs a
        weaker sentence, never a false discovery claim."""
        na, nb = self.docid_to_node.get(a), self.docid_to_node.get(b)
        if not (na and nb):
            return False
        ta = " ".join((self.label.get(na) or "").lower().split())
        tb = " ".join((self.label.get(nb) or "").lower().split())
        return bool(ta) and ta == tb

    def _title(self, doc_id: str) -> str:
        nid = self.docid_to_node.get(doc_id)
        t = self.label.get(nid, doc_id) if nid else doc_id
        return t if len(t) <= 60 else t[:59] + "…"

    # -- the guarantee ------------------------------------------------------
    def verify(self, explanation: dict) -> dict:
        """Re-check every fact's evidence against the real edge set.

        A fact whose triple is not in the graph is a fabrication; this is what makes
        "nothing is invented" a checked property rather than a promise.
        """
        total = bad = 0
        # Every section, explicitly. A verifier that silently skips a section is
        # worse than none: it reports ok=True over facts it never looked at, which
        # is exactly the shape of the bug it exists to catch.
        sections = [list(explanation["per_doc"].values()),
                    [explanation.get("pairs", [])],
                    [explanation.get("space", [])]]
        for group in sections:
            for facts in group:
                for f in facts:
                    for e in f["evidence"]:
                        total += 1
                        if tuple(e) not in self._edges:
                            bad += 1
        return {"evidence_triples": total, "not_in_graph": bad, "ok": bad == 0}
