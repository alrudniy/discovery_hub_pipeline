#!/usr/bin/env python3
"""
baseline_check.py -- is the R-GCN's AUC actually better than a trivial baseline?

WHY THIS EXISTS
06_train_rgcn.py reports roc_auc=0.813 on held-out edges and nothing compares that
to a scorer that learned nothing. It needs comparing, because of HOW the metric
samples negatives:

    graph_eval.sample_negatives() draws u and v UNIFORMLY over all 1.6M nodes.

A real edge joins two nodes that have edges at all. A uniform random pair usually
joins two near-isolated nodes (mean degree ~6 over 1.6M nodes, heavy-tailed). So
ANY score correlated with node degree separates positives from negatives without
learning a thing. "Popular nodes connect to popular nodes" would post a high AUC
on this protocol while knowing nothing about the graph.

WHAT THIS DOES
Runs every scorer through ONE harness with the SAME split (seed=config.SEED), the
SAME sampled negatives, and the SAME candidate draws, so the rows are comparable:

  random        pseudo-random scores            -- sanity: must land at ~0.5
  tail_degree   deg(v)                          -- pure popularity. Learns nothing.
  pref_attach   deg(u)*deg(v)                   -- preferential attachment.
  common_neigh  |N(u) & N(v)|                   -- classic structural heuristic
  adamic_adar   sum 1/log(deg(w)) for w in N(u)&N(v)
  rgcn          Z[u] . Z[v]                     -- the trained embedding

Degrees/adjacency come from MESSAGE-PASSING EDGES ONLY -- the same edges the R-GCN
saw. Using all edges would leak the test set into the baseline.

The rgcn row re-runs through this harness rather than being copied from
rgcn_eval.json, which also validates the harness reproduces the reported number.

NOTHING IS FABRICATED. Every number here is computed from the real graph.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from discovery_hub import config
from discovery_hub import graph_eval


def load_graph():
    nodes = [json.loads(l) for l in (config.GRAPH_DIR / "nodes.jsonl").open() if l.strip()]
    edges = [json.loads(l) for l in (config.GRAPH_DIR / "edges.jsonl").open() if l.strip()]
    node_ids = [n["node_id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    return nodes, edges, node_ids, idx


def evaluate_scorer(score_fn, pos_pairs, pos_set, n_nodes, seed, k=10,
                    ranking_negs=50, ranking_sample=50000):
    """
    Mirror of graph_eval.evaluate_link_prediction, but scoring is pluggable.

    The rng is re-seeded per scorer and consumed in the same order, so every
    scorer sees byte-identical negatives and candidate lists. AUC uses all
    positives; the ranking metrics stop at `ranking_sample` pairs (same prefix
    for every scorer) because the loop is O(n_pairs * ranking_negs) in Python.
    """
    rng = np.random.default_rng(seed)

    su = np.array([u for u, _ in pos_pairs])
    sv = np.array([v for _, v in pos_pairs])
    pos_scores = np.asarray(score_fn(su, sv), dtype=float)

    neg = graph_eval.sample_negatives(len(pos_pairs), n_nodes, pos_set, rng)
    nu = np.array([u for u, _ in neg])
    nv = np.array([v for _, v in neg])
    neg_scores = np.asarray(score_fn(nu, nv), dtype=float)
    auc = graph_eval.roc_auc(pos_scores, neg_scores)

    hits, rr, cnt = 0, 0.0, 0
    for j, (u, v) in enumerate(pos_pairs):
        if j >= ranking_sample:
            break
        cand = [v]
        t = 0
        while len(cand) < ranking_negs + 1 and t < (ranking_negs + 1) * 20:
            w = int(rng.integers(0, n_nodes))
            t += 1
            if w != u and (u, w) not in pos_set:
                cand.append(w)
        cand_arr = np.array(cand)
        scores = np.asarray(score_fn(np.full(len(cand_arr), u), cand_arr), dtype=float)
        rank = 1 + int(np.sum(scores[1:] > scores[0]))   # ties => optimistic, as in graph_eval
        hits += 1 if rank <= k else 0
        rr += 1.0 / rank
        cnt += 1

    return {"roc_auc": auc, f"hits@{k}": hits / cnt if cnt else 0.0,
            "mrr": rr / cnt if cnt else 0.0, "num_auc_pairs": len(pos_pairs),
            "num_ranking_pairs": cnt}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--ranking-sample", type=int, default=50000,
                    help="pairs used for hits@10/mrr (AUC always uses all)")
    ap.add_argument("--out", default="./rgcn_baseline.json")
    args = ap.parse_args()

    print("loading graph ...")
    nodes, edges, node_ids, idx = load_graph()
    n = len(node_ids)
    mp_edges, val_edges, test_edges = graph_eval.split_edges(
        edges, val_frac=args.val_frac, test_frac=args.test_frac, seed=config.SEED)
    print(f"{n:,} nodes / {len(edges):,} edges | "
          f"split mp={len(mp_edges):,} val={len(val_edges):,} test={len(test_edges):,}")

    test_pairs = graph_eval.edges_to_pairs(test_edges, idx)
    all_pos = set(graph_eval.edges_to_pairs(edges, idx))

    # ---- structure from MESSAGE-PASSING EDGES ONLY (no test leakage) ----
    print("building degree + adjacency from message-passing edges only ...")
    deg = np.zeros(n, dtype=np.float64)
    adj = defaultdict(set)
    for e in mp_edges:
        s, d = idx.get(e["src"]), idx.get(e["dst"])
        if s is None or d is None:
            continue
        deg[s] += 1
        deg[d] += 1
        adj[s].add(d)
        adj[d].add(s)
    print(f"  degree: mean {deg.mean():.2f}  median {np.median(deg):.0f}  "
          f"max {deg.max():.0f}  zero-degree nodes {int((deg == 0).sum()):,} "
          f"({(deg == 0).mean():.1%})")

    inv_log_deg = np.zeros(n, dtype=np.float64)
    nz = deg > 1
    inv_log_deg[nz] = 1.0 / np.log(deg[nz])

    # ---- scorers ----
    def s_random(u, v):
        return np.random.default_rng(12345).random(len(u))

    def s_tail_degree(u, v):
        return deg[v]

    def s_pref_attach(u, v):
        return deg[u] * deg[v]

    def s_common_neigh(u, v):
        return np.array([len(adj[int(a)] & adj[int(b)]) for a, b in zip(u, v)],
                        dtype=float)

    def s_adamic_adar(u, v):
        out = np.empty(len(u), dtype=float)
        for i, (a, b) in enumerate(zip(u, v)):
            common = adj[int(a)] & adj[int(b)]
            out[i] = float(inv_log_deg[list(common)].sum()) if common else 0.0
        return out

    scorers = [("random", s_random),
               ("tail_degree", s_tail_degree),
               ("pref_attach", s_pref_attach),
               ("common_neigh", s_common_neigh),
               ("adamic_adar", s_adamic_adar)]

    emb_p = config.ARTIFACT_DIR / "rgcn_node_emb.npy"
    if emb_p.exists():
        Z = np.load(emb_p)
        ids = json.loads((config.ARTIFACT_DIR / "node_ids.json").read_text())
        if ids != node_ids:
            print("  WARNING: node_ids.json order != nodes.jsonl order; remapping")
            row = {nid: i for i, nid in enumerate(ids)}
            perm = np.array([row[nid] for nid in node_ids])
            Z = Z[perm]
        scorers.append(("rgcn", lambda u, v: np.sum(Z[u] * Z[v], axis=1)))
    else:
        print(f"  NOTE: {emb_p} absent -- baselines only, no rgcn row.")

    results = {}
    for name, fn in scorers:
        t0 = time.time()
        print(f"\nscoring {name} ...")
        r = evaluate_scorer(fn, test_pairs, all_pos, n, config.SEED,
                            ranking_sample=args.ranking_sample)
        r["seconds"] = round(time.time() - t0, 1)
        results[name] = r
        print(f"  AUC {r['roc_auc']:.4f}  hits@10 {r['hits@10']:.4f}  "
              f"mrr {r['mrr']:.4f}  ({r['seconds']}s)")

    print("\n" + "=" * 72)
    print("LINK PREDICTION: TRAINED R-GCN vs TRIVIAL BASELINES")
    print("negatives sampled uniformly over all nodes (graph_eval.sample_negatives)")
    print("=" * 72)
    print(f"{'scorer':16s} {'AUC':>8s} {'hits@10':>9s} {'MRR':>8s}   {'learns?':>8s}")
    print("-" * 72)
    learns = {"random": "no", "tail_degree": "no", "pref_attach": "no",
              "common_neigh": "no", "adamic_adar": "no", "rgcn": "YES"}
    for name, r in results.items():
        print(f"{name:16s} {r['roc_auc']:8.4f} {r['hits@10']:9.4f} {r['mrr']:8.4f}   "
              f"{learns.get(name, '?'):>8s}")
    print("-" * 72)

    if "rgcn" in results:
        rg = results["rgcn"]["roc_auc"]
        best_triv = max((r["roc_auc"], nm) for nm, r in results.items() if nm != "rgcn")
        margin = rg - best_triv[0]
        print(f"\nR-GCN AUC {rg:.4f} vs best trivial ({best_triv[1]}) {best_triv[0]:.4f}"
              f"  ->  margin {margin:+.4f}")
        if margin <= 0:
            print("VERDICT: the R-GCN does NOT beat a scorer that learned nothing.")
            print("         The reported AUC is an artifact of uniform negative sampling.")
        elif margin < 0.05:
            print("VERDICT: the R-GCN barely beats a trivial baseline. The headline AUC")
            print("         mostly measures degree, not learned structure.")
        else:
            print("VERDICT: the R-GCN beats every trivial baseline by a real margin.")

    payload = {"protocol": "graph_eval.evaluate_link_prediction (uniform negatives)",
               "seed": config.SEED, "num_nodes": n, "num_edges": len(edges),
               "num_test_edges": len(test_pairs),
               "ranking_negatives": 50,
               "structure_from": "message-passing edges only (no test leakage)",
               "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
