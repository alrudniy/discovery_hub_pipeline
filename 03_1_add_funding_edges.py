#!/usr/bin/env python3
"""
03_1_add_funding_edges.py  --  add SBIR funding to the graph, ADDITIVELY.

Why this exists: `parse_sbir` drops the award amount, so funding never reaches
DiscoveryDoc and therefore never reaches the graph. The "correct" fix (carry
Award Amount through convert_sbir -> parse_sbir -> DiscoveryDoc.extra) requires
REGENERATING normalized/docs.jsonl -- which would invalidate the doc order that
doc_vectors.npy / doc_ids.json / the FAISS index are all aligned to. Never do that
between an embed and an index rebuild.

So instead this reads the raw SBIR bulk CSV directly, matches awards to the
`technology:sbir:<award_id>` nodes already in the graph, and emits:

    (:technology)-[:funded_by {amount, year}]->(:funder)

Nothing else is touched. By default it writes to a NEW directory so the original
graph stays byte-identical; pass --in-place only when you are sure.

CAUTION: this introduces a new node type ("funder") and relation ("funded_by").
06_train_rgcn.py may iterate a fixed RELATIONS list -- check it before training an
R-GCN on the augmented graph, or it will silently ignore (or choke on) the new rel.

Usage:
  python 03_1_add_funding_edges.py                       # -> data/graph_funded/
  python 03_1_add_funding_edges.py --out data/graph_v2
  python 03_1_add_funding_edges.py --in-place            # overwrite data/graph/
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from discovery_hub import config

C_AGENCY, C_AMOUNT, C_YEAR = "Agency", "Award Amount", "Award Year"
C_CONTRACT, C_TRACKING = "Contract", "Agency Tracking Number"
C_FIRM, C_TITLE = "Company", "Award Title"


def norm_entity(s: str) -> str:
    """Same normalization 03_build_graph uses to dedup entity labels."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def make_award_id(row: dict) -> str:
    """Must match 01_6_convert_sbir.make_award_id exactly, or nothing joins."""
    aid = (row.get(C_CONTRACT) or "").strip() or (row.get(C_TRACKING) or "").strip()
    if not aid:
        seed = (row.get(C_FIRM, "") + row.get(C_TITLE, "") + row.get(C_YEAR, ""))
        aid = "h" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
    return re.sub(r"\s+", "_", aid)


def parse_amount(s: str) -> float | None:
    s = re.sub(r"[^0-9.]", "", (s or ""))       # strip $ and commas
    try:
        return float(s) if s else None
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", default=None, help="input graph (default DH graph dir)")
    ap.add_argument("--sbir-csv", default=None,
                    help="raw SBIR bulk CSV (default $DH_DATA_ROOT/raw/sbir_bulk.json)")
    ap.add_argument("--out", default=None, help="output dir (default <graph-dir>_funded)")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input graph")
    args = ap.parse_args()

    gdir = Path(args.graph_dir) if args.graph_dir else config.GRAPH_DIR
    csv_path = Path(args.sbir_csv) if args.sbir_csv else config.RAW_DIR / "sbir_bulk.json"
    out = gdir if args.in_place else (Path(args.out) if args.out
                                      else gdir.parent / (gdir.name + "_funded"))
    for p in (gdir / "nodes.jsonl", gdir / "edges.jsonl"):
        if not p.exists():
            print(f"ERROR: {p} not found -- run 03_build_graph.py first.", file=sys.stderr)
            return 2
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        return 2

    nodes = {}
    for line in (gdir / "nodes.jsonl").read_text().splitlines():
        if line.strip():
            nd = json.loads(line)
            nodes[nd["node_id"]] = nd
    edges = [json.loads(l) for l in (gdir / "edges.jsonl").read_text().splitlines() if l.strip()]
    n_nodes0, n_edges0 = len(nodes), len(edges)
    print(f"loaded graph: {n_nodes0} nodes, {n_edges0} edges from {gdir}")

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2 ** 31 - 1)

    seen_edge = set()
    matched = no_agency = no_amount = unmatched = 0
    total_funding = 0.0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            tech_id = f"technology:sbir:{make_award_id(row)}"
            if tech_id not in nodes:            # award filtered out of the corpus
                unmatched += 1
                continue
            agency = (row.get(C_AGENCY) or "").strip()
            if not agency:
                no_agency += 1
                continue
            amount = parse_amount(row.get(C_AMOUNT, ""))

            fid = f"funder:{norm_entity(agency)}"
            if fid not in nodes:
                nodes[fid] = {"node_id": fid, "ntype": "funder", "label": agency}
            key = (tech_id, fid)
            if key in seen_edge:                # one award, one edge (CSV has dupes)
                continue
            seen_edge.add(key)
            if amount is None:                  # count only edges we actually add
                no_amount += 1
            else:
                total_funding += amount
            e = {"src": tech_id, "dst": fid, "rel": "funded_by"}
            if amount is not None:
                e["amount"] = amount
            yr = (row.get(C_YEAR) or "").strip()
            if yr:
                e["year"] = yr
            edges.append(e)
            matched += 1

    out.mkdir(parents=True, exist_ok=True)
    with (out / "nodes.jsonl").open("w") as fh:
        for nd in nodes.values():
            fh.write(json.dumps(nd) + "\n")
    with (out / "edges.jsonl").open("w") as fh:
        for e in edges:
            fh.write(json.dumps(e) + "\n")

    ntypes, rels = {}, {}
    for nd in nodes.values():
        ntypes[nd["ntype"]] = ntypes.get(nd["ntype"], 0) + 1
    for e in edges:
        rels[e["rel"]] = rels.get(e["rel"], 0) + 1
    meta = {"num_nodes": len(nodes), "num_edges": len(edges),
            "nodes_by_type": ntypes, "edges_by_rel": rels,
            "funding": {"funded_edges": matched,
                        "funders": ntypes.get("funder", 0),
                        "total_award_usd": round(total_funding, 2)}}
    with (out / "graph_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nfunding: matched {matched} awards -> {ntypes.get('funder', 0)} funders")
    print(f"  unmatched CSV rows (not in corpus): {unmatched}")
    print(f"  missing agency: {no_agency} | missing amount: {no_amount}")
    print(f"  total award value: ${total_funding:,.0f}")
    print(f"\ngraph: {n_nodes0} -> {len(nodes)} nodes, {n_edges0} -> {len(edges)} edges")
    print(f"  nodes by type: {ntypes}")
    print(f"  edges by rel : {rels}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
