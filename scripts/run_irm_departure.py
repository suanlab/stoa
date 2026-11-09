#!/usr/bin/env python3
"""H1: how far KV traces depart from the independent reference model (§BF; ADVISE H1).

Buffer-management theory from A0 to LRU-K assumes each page is referenced as a stationary process. If
reuse is a stationary Poisson process per block, a block's count in the prefix k and in an equal-length
future window Y satisfy Robbins' identity E[Y | k] = (k + 1) n_{k+1} / n_k, with n_k the number of
blocks seen k times; and Good-Turing's missing mass n_1 / N estimates the chance that the next access
is to a block not seen before. Both need no model, only the counts. This script compares them with
what happens after the split (0.5, equal halves by request), for k = 1..5:

  rho_k   = actual mean future count of blocks seen k times / Robbins' prediction
  gt      = Good-Turing missing mass vs the actual share of post-split accesses to unseen blocks

Control: the same trace with its requests in random order (seed 0), which makes the reference stream
exchangeable; there rho_k should be near 1. Pre-registered (ADVISE H1): rejected if two or more of the
four Alibaba traces have rho_1 >= 0.8; expected below 0.5 for every k <= 5 on production.

What the test can and cannot say (review 2026-11-09, R4): Robbins' identity needs a stationary POISSON
(independent reference) stream, not stationarity alone; a stationary but serially dependent stream can
fail it too. So a departure here rejects the independent reference model, and the shuffle removes
session structure, bursts and drift at once. Two further controls (§BJ), Mooncake only:

  shuffle seeds   the full shuffle with seeds 0-19: the spread of rho_k and of the Good-Turing gap
  chunk shuffles  requests cut into consecutive chunks of C requests (C = 100, 1000, 3000), chunks put in
                  random order (seed 0), order kept inside each chunk: structure shorter than a chunk
                  survives, drift across the trace does not

Registered before running the chunk shuffles: the departure is local structure (sessions, bursts) if
rho_1 stays below 0.5 with 1000-request chunks on both production traces; it is drift across the trace
if rho_1 reaches 0.8 there; in between, neither.
Usage: python3 scripts/run_irm_departure.py [--bailian]
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import stoa.mooncake as mc  # noqa: E402
from stoa.traces import AccessTrace, Workload  # noqa: E402

MOONCAKE = {n: f"data/mooncake_{n}_trace.jsonl" for n in ("conversation", "toolagent", "synthetic")}
BAILIAN = {n: f"data/bailian_{n}_mooncake.jsonl" for n in ("traceA", "traceB", "coder", "thinking")}
K = (1, 2, 3, 4, 5)
SEEDS = tuple(range(20))
CHUNKS = (100, 1000, 3000)


def shuffled(wl: Workload, seed: int = 0) -> Workload:
    """Requests in random order: every block's accesses move with their request, so counts per block
    are unchanged and only the order (hence stationarity) changes."""
    steps = sorted({t for t, _ in wl.accesses})
    perm = dict(zip(steps, random.Random(seed).sample(steps, len(steps))))
    acc = sorted((perm[t], i) for t, i in wl.accesses)
    tr = {}
    for t, i in acc:
        tr.setdefault(i, []).append(t)
    return Workload(items=wl.items, accesses=acc,
                    traces={i: AccessTrace(item_id=i, access_steps=s) for i, s in tr.items()})


def chunk_shuffled(wl: Workload, chunk: int, seed: int = 0) -> Workload:
    """Consecutive runs of `chunk` requests moved as units into random order; order inside a run kept."""
    steps = sorted({t for t, _ in wl.accesses})
    chunks = [steps[i:i + chunk] for i in range(0, len(steps), chunk)]
    random.Random(seed).shuffle(chunks)
    perm = dict(zip([t for c in chunks for t in c], steps))
    acc = sorted((perm[t], i) for t, i in wl.accesses)
    tr = {}
    for t, i in acc:
        tr.setdefault(i, []).append(t)
    return Workload(items=wl.items, accesses=acc,
                    traces={i: AccessTrace(item_id=i, access_steps=s) for i, s in tr.items()})


def analyse(wl: Workload) -> dict:
    st = max(1, wl.time_split(0.5))
    pre, fut = wl.prefix_stats(st), wl.future_counts(st)
    vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
    k_of = {b: int(pre[b]["count"]) for b in vis}
    n = Counter(k_of.values())
    N = sum(k * v for k, v in n.items())
    # the future count Robbins speaks of counts every access in the window, so use all accesses
    # after the split for visible blocks (a visible block has no creation after the split)
    rows = []
    for k in K:
        blocks = [b for b in vis if k_of[b] == k]
        if not blocks or not n[k + 1]:
            continue
        actual = sum(fut.get(b, 0) for b in blocks) / len(blocks)
        robbins = (k + 1) * n[k + 1] / n[k]
        rows.append({"k": k, "n_k": n[k], "actual_mean_future": round(actual, 4),
                     "robbins": round(robbins, 4), "rho": round(actual / robbins, 4),
                     "p_dead": round(sum(1 for b in blocks if fut.get(b, 0) == 0) / len(blocks), 4)})
    post = [(t, i) for t, i in wl.accesses if t >= st]
    new_share = sum(1 for _, i in post if wl.traces[i].access_steps[0] >= st) / len(post)
    return {"split_step": st, "good_turing_missing_mass": round(n[1] / N, 4),
            "post_split_access_share_new": round(new_share, 4), "by_k": rows}


def main() -> int:
    bail = "--bailian" in sys.argv
    traces = BAILIAN if bail else MOONCAKE
    if bail:
        mc.BLOCK_TOKENS = 16
    out = {"config": {"split": 0.5, "k": list(K), "control": "requests shuffled, seed 0",
                      "shuffle_seeds": list(SEEDS), "chunks_requests": list(CHUNKS),
                      "registered": "rejected if >= 2 of 4 Alibaba traces have rho_1 >= 0.8"}, "traces": {}}
    for name, path in traces.items():
        wl = mc.load_mooncake(str(ROOT / path))
        rec = {"as_recorded": analyse(wl)}
        if not bail or name == "traceA":
            rec["shuffled_control"] = analyse(shuffled(wl))
        if not bail:
            per_seed = [analyse(shuffled(wl, seed)) for seed in SEEDS]
            rho = {k: [r["rho"] for a in per_seed for r in a["by_k"] if r["k"] == k] for k in K}
            gap = [round(a["good_turing_missing_mass"] - a["post_split_access_share_new"], 4) for a in per_seed]
            rec["shuffle_seeds"] = {"seeds": list(SEEDS),
                                    "rho_range": {str(k): [min(v), max(v)] for k, v in rho.items() if v},
                                    "gt_minus_actual_range": [min(gap), max(gap)]}
            rec["chunk_shuffles"] = {str(c): analyse(chunk_shuffled(wl, c)) for c in CHUNKS}
            print(name, "seeds", rec["shuffle_seeds"]["rho_range"], "GT gap", rec["shuffle_seeds"]["gt_minus_actual_range"],
                  "chunks", {c: [(r["k"], r["rho"]) for r in v["by_k"]][:2] for c, v in rec["chunk_shuffles"].items()},
                  flush=True)
        out["traces"][name] = rec
        for kk, v in rec.items():
            if kk in ("shuffle_seeds", "chunk_shuffles"):
                continue
            print(name, kk, "GT", v["good_turing_missing_mass"], "new share", v["post_split_access_share_new"],
                  [(r["k"], r["rho"]) for r in v["by_k"]], flush=True)
    dest = ROOT / "experiments" / ("irm_departure_bailian.json" if bail else "irm_departure.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
