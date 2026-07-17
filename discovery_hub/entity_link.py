"""
Query -> graph entity linking: the part that makes the graph signal *real*.

The old graph signal scored candidates against the centroid of the top TEXT hits
-- so it was downstream of text retrieval and largely re-encoded the same signal,
which is why the eval (stage 10) found it didn't help. This module builds the
graph-space query from the query's OWN named entities instead: if a query names
an organization, inventor, or facility that exists in the graph, we anchor on
that node and let the R-GCN's learned structure (who invented/assigned/affiliated
with what) surface connected technologies -- including ones whose text does not
lexically match the query. That is the actual promise of the knowledge graph.

Honesty by construction: if a query links to NO graph entity (common for novel,
register-crossing research interests that don't name known players), the signal
ABSTAINS rather than guessing. It can only help or stay silent, never inject
noise. That also scopes the deck claim honestly -- the graph helps when the query
names entities connected to the answer, not on every query.

Pure, deterministic, unit-tested (tests/test_hybrid.py).
"""
from __future__ import annotations

import re

# Entity node types worth linking a free-text query to. Technologies are the
# retrieval targets, not query anchors, so they are excluded.
LINKABLE_NTYPES = ("organization", "inventor", "expert", "facility")

_NORM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
# Tokens too generic to alias a multi-word entity on (would cause false links).
_STOP_ALIAS = {"university", "medical", "center", "institute", "inc", "corp",
               "llc", "ltd", "company", "co", "the", "of", "and", "school",
               "college", "hospital", "labs", "laboratory", "laboratories"}

# Trailing legal-entity markers. Stripping these is what turns "Genentech Inc"
# into the name "genentech"; a label that still has >1 token afterwards is a
# phrase, not a name, and none of its words may alias it.
_LEGAL_SUFFIX = {
    "inc", "incorporated", "llc", "lllp", "llp", "lp", "ltd", "limited",
    "corp", "corporation", "co", "company", "companies", "gmbh", "mbh", "ag",
    "kg", "kgaa", "se", "nv", "bv", "cv", "sa", "sas", "sarl", "srl", "spa",
    "ab", "as", "asa", "oy", "oyj", "aps", "plc", "pte", "pty", "kk", "ug",
    "bvba", "nuel", "sl", "slu", "sro", "zoo", "dd", "doo", "ou", "sca",
    "trust", "holding", "holdings", "group", "intl", "international", "usa",
}


def normalize(s: str) -> str:
    return _WS.sub(" ", _NORM.sub(" ", (s or "").lower())).strip()


def _strip_legal_suffix(toks: list[str]) -> list[str]:
    """Drop the trailing run of legal-entity markers: ['genentech','inc'] -> ['genentech']."""
    out = list(toks)
    while out and out[-1] in _LEGAL_SUFFIX:
        out.pop()
    return out


# Zipf frequency above which a would-be alias is an ordinary English word rather
# than a company name. Calibrated, not guessed -- measured on this graph's own
# aliases and the names they must not eat:
#
#   REJECT >= 3.5 : science 5.12  technology 5.09  base 5.05  album 5.02
#                   approach 4.93  holding 4.89  select 4.45  tablet 4.05
#                   pipeline 4.04  signaling 3.65
#   KEEP   <  3.5 : pfizer 3.11  novartis 2.58  astrazeneca 2.43  genentech 2.18
#                   moderna 2.23  biogen 1.98  regeneron 1.72  medimmune 1.44
#                   amplimmune 0.00
#
# 3.11 (pfizer) to 3.65 (signaling) is the whole margin, so this is a real but
# narrow separation -- the reason it works at all is that it runs AFTER the
# structural rule, which has already removed the fragments ("tablet" out of "hot
# album tansansen tablet inc") that no frequency signal could separate. Two other
# signals were measured and REJECTED as non-separating: corpus idf (norwalk 10.2
# and album 9.7 are RARER than yale 6.4 and genentech 5.5) and entity-surface
# frequency (kinase/autologous/select appear in 1-4 surfaces, as does genentech).
_MAX_ALIAS_ZIPF = 3.5


def _english_word_filter(max_zipf: float | None):
    """Return is_common_word(token) -> bool. Fails loudly rather than silently
    degrading: a filter that quietly becomes a no-op is how build_surface_index
    got here in the first place."""
    if max_zipf is None:
        return lambda t: False
    try:
        from wordfreq import zipf_frequency
    except ImportError as e:  # noqa: F841
        raise ImportError(
            "build_surface_index(max_alias_zipf=...) needs `wordfreq` (pip install "
            "wordfreq). Pass max_alias_zipf=None to disable the English-word filter "
            "DELIBERATELY -- but know that it re-admits aliases like 'base' (from "
            "'BASE SE') and 'technology' (from 'TECHNOLOGY HOLDING LLC').")
    return lambda t: zipf_frequency(t, "en") >= max_zipf


def build_surface_index(nodes, *, min_surface_chars: int = 3,
                        min_alias_chars: int = 4,
                        max_alias_zipf: float | None = _MAX_ALIAS_ZIPF,
                        linkable: tuple[str, ...] = LINKABLE_NTYPES
                        ) -> dict[str, set[str]]:
    """
    surface form -> set(node_id) for entity nodes. Indexes the full normalized
    label, plus a single-token alias ONLY when that token is the entity's whole
    name once legal suffixes are stripped ("Genentech Inc" -> "genentech").

    WHY NOT UNIQUENESS (the previous rule, and why it failed)
    The old rule promoted ANY token of a multi-word label that was globally unique
    among surfaces. Uniqueness is precisely the wrong criterion: in an index of
    ~926k surfaces the globally-unique tokens are the junk ones by construction,
    because a distinctive company name is often SHARED across surfaces while an
    odd word appears once. Measured consequences on the judged slice:

        "CXCR4 gene therapy approach"        -> organization:science approach
        "Oral tablet formulation for CML"    -> organization:hot album tansansen tablet inc
        "Drugs targeting glutamate signaling"-> organization:cell signaling technology inc
        "Pipeline candidates for ... pain"   -> organization:pipeline therapeutics inc

    Each is one common word ("approach", "tablet", "signaling", "pipeline") that
    happened to be unique. The graph then averaged embeddings of unrelated
    entities, and its top-100 contained ZERO relevant docs across 62 firing
    queries (median best rank 195,898 of 600,738). The docstring claimed precision
    over recall; the mechanism delivered the opposite.

    THE RULE NOW: a fragment is never an alias. A token is promoted only if
    stripping trailing legal-entity suffixes leaves exactly that one token -- i.e.
    it IS the entity's complete name, not a piece of it. "science approach" has no
    legal suffix and two tokens, so neither word aliases it. Kept from the old
    rule: global uniqueness among aliases, the stopword list, and a length floor
    (raised to 4).

    NOT USED -- rejected on evidence: a corpus-idf ("common English word") filter.
    Measured on this corpus it does not separate junk from names -- 'norwalk'
    (idf 10.2) and 'album' (9.7) are RARER than 'yale' (6.4) and 'genentech'
    (5.5), so every threshold that rejects 'tablet' (5.4) also rejects 'pfizer'
    (5.2). Rarity is not the signal; structure is.

    Known cost: "Eli Lilly" is no longer reachable as "lilly" (two tokens, no
    legal suffix). That is the precision-first trade the docstring always claimed
    to make. Full surfaces shorter than `min_surface_chars` are dropped, which is
    what stops expert:m from linking the "M" in "Anti-oncostatin M antibody".
    """
    is_common = _english_word_filter(max_alias_zipf)
    full: dict[str, set[str]] = {}
    token_owners: dict[str, set[str]] = {}   # token -> node_ids it could alias

    for nd in nodes:
        if nd.get("ntype") not in linkable:
            continue
        nid = nd["node_id"]
        surf = normalize(nd.get("label") or nid.split(":", 1)[-1])
        if len(surf) < min_surface_chars:    # "m", "ab" -- false-link magnets
            continue
        full.setdefault(surf, set()).add(nid)
        toks = surf.split()
        if len(toks) > 1:
            core = _strip_legal_suffix(toks)
            if len(core) == 1:               # the label IS this token + "Inc"/"LLC"
                t = core[0]
                if (t not in _STOP_ALIAS and len(t) >= min_alias_chars
                        and not is_common(t)):
                    token_owners.setdefault(t, set()).add(nid)

    index = dict(full)
    for tok, owners in token_owners.items():
        if tok in index:            # already a full surface; don't override
            continue
        if len(owners) == 1:        # unique alias only
            index[tok] = set(owners)
    return index


def link_query(query: str, surface_index: dict[str, set[str]],
               max_gram: int = 4) -> list[str]:
    """
    Greedy longest-match n-gram linking. Scans left to right; at each position
    tries the longest n-gram (up to max_gram) that is a known surface form, links
    it, and advances past it. Returns the sorted unique list of linked node_ids.
    """
    toks = normalize(query).split()
    linked: set[str] = set()
    i = 0
    while i < len(toks):
        matched = False
        for g in range(min(max_gram, len(toks) - i), 0, -1):
            span = " ".join(toks[i:i + g])
            if span in surface_index:
                linked.update(surface_index[span])
                i += g
                matched = True
                break
        if not matched:
            i += 1
    return sorted(linked)
