#!/usr/bin/env python3
"""The split-time ceiling is set by where it puts blocks that will never be read again (§AZ).

`attainable_ceiling` gives every block visible at the split its true future count and every new block
0, then runs the greedy placement (belief descending, stable in item order, fastest tier with room on
ties). A visible block with NO future access has belief 0, exactly like a new block, and every pre-split
block precedes every new block in item order -- so the ceiling puts dead visible blocks into the fast
tiers ahead of the blocks that arrive after the split. Both placements cost a dead block nothing; a
policy that knows the visible futures (which the ceiling grants) loses nothing by parking them last.

An independent review (2026-10-25, P0-2) asked whether the ceiling is an upper bound for its policy
class. This script answers on two levels:
  traces   ceiling as published vs the same with dead visible blocks ordered after every other block
           (`dead-last`), both action spaces, all three Mooncake traces
  toys     200 random 10-block workloads (6 visible, 4 new; uniform size): the published ceiling and
           dead-last against an exhaustive search over every capacity-feasible placement of the visible
           blocks, new blocks then taking the fastest tier with room in creation order
Captured share = (heuristic - ceiling) / (heuristic - oracle), as everywhere in §V-B.
Usage: python3 scripts/run_ceiling_ties.py
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_arrival_admission import TRACES  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa.environment import ItemState, Representation, Timing  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS, weighted_cost  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402
from stoa.traces import AccessTrace, Workload  # noqa: E402

ALL = list(sq.LEGAL_ACTIONS)
TIER = [a for a in ALL if a.representation == Representation.PLAINTEXT and a.timing == Timing.ONLINE]
SPACES = {"all": ALL, "tier": TIER}


def dead_last(wl: Workload, dead: set) -> Workload:
    """The same workload with dead visible blocks moved to the end of item order: on the belief-0 tie
    the greedy pass then reaches them after every new block."""
    return dataclasses.replace(wl, items=[i for i in wl.items if i.item_id not in dead]
                               + [i for i in wl.items if i.item_id in dead])


def split_sets(wl: Workload):
    st = max(1, wl.time_split(0.5))
    fut = wl.future_counts(st)
    vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
    return st, fut, vis, {b for b in vis if fut.get(b, 0) == 0}


def toy(rng: random.Random, n_vis: int = 6, n_new: int = 4, split: int = 50) -> Workload:
    s = {}
    for i in range(n_vis):
        s[f"v{i}"] = sorted(rng.sample(range(0, split), rng.randint(1, 3))
                            + rng.sample(range(split, 100), rng.randint(0, 6)))
    for i in range(n_new):
        s[f"n{i}"] = sorted(rng.sample(range(split, 100), rng.randint(1, 6)))
    return Workload(items=[ItemState(item_id=i, size_bytes=1) for i in s],
                    accesses=sorted((t, i) for i, v in s.items() for t in v),
                    traces={i: AccessTrace(item_id=i, access_steps=v) for i, v in s.items()})


def exhaustive(wl: Workload, actions, sim, w) -> float:
    """Min true cost over every feasible tier assignment of the visible blocks (each at its cheapest
    action in that tier for its true count); new blocks then by the default rule."""
    st, fut, vis_ids, _ = split_sets(wl)
    vis = [i for i in wl.items if i.item_id in set(vis_ids)]
    new = [i for i in wl.items if i.item_id not in set(vis_ids)]
    caps = sim.tier_capacities(wl.total_bytes)
    tiers = sorted({a.tier for a in actions}, key=sim.read_latency)

    def cost(it, t, n_belief, n_true):
        a = min((a for a in actions if a.tier == t),
                key=lambda a: (weighted_cost(sim.cost_of_placement(a, n_belief, it.representation), w)))
        return weighted_cost(sim.cost_of_placement(a, n_true, it.representation), w)
    best = float("inf")
    for asg in itertools.product(tiers, repeat=len(vis)):
        rem, c, ok = dict(caps), 0.0, True
        for it, t in zip(vis, asg):
            if rem[t] < it.size_bytes:
                ok = False
                break
            rem[t] -= it.size_bytes
            c += cost(it, t, fut[it.item_id], fut[it.item_id])
        if not ok:
            continue
        for it in new:
            t = next(t for t in tiers if rem[t] >= it.size_bytes)
            rem[t] -= it.size_bytes
            c += cost(it, t, 0, fut[it.item_id])
        best = min(best, c)
    return best


def main() -> int:
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    rows, toys = [], {}
    try:
        for space, acts in SPACES.items():
            sq.LEGAL_ACTIONS = acts
            rng = random.Random(0)
            n_pub = n_dl = 0
            gap_pub, gap_dl = [], []
            for _ in range(200):
                wl = toy(rng)
                _, _, vis, dead = split_sets(wl)
                ex = exhaustive(wl, acts, sim, w)
                pub, dl = sq.attainable_ceiling(wl, vis), sq.attainable_ceiling(dead_last(wl, dead), vis)
                assert pub >= ex - 1e-9 and dl >= ex - 1e-9, "exhaustive search missed a placement"
                assert dl <= pub + 1e-9, "parking dead blocks last made the ceiling worse"
                if pub > ex + 1e-9:
                    n_pub += 1
                    gap_pub.append((pub - ex) / ex)
                if dl > ex + 1e-9:
                    n_dl += 1
                    gap_dl.append((dl - ex) / ex)
            toys[space] = {"n": 200, "published_above_exhaustive": n_pub, "dead_last_above_exhaustive": n_dl,
                           "max_rel_gap_published": round(max(gap_pub, default=0.0), 4),
                           "max_rel_gap_dead_last": round(max(gap_dl, default=0.0), 4)}
            print(space, toys[space], flush=True)
        for name, path in TRACES.items():
            wl = load_mooncake(str(ROOT / path))
            st, fut, vis, dead = split_sets(wl)
            pre = wl.prefix_stats(st)
            for space, acts in SPACES.items():
                sq.LEGAL_ACTIONS = acts
                heur = sq.place_with_beliefs(wl, {i.item_id: float(pre[i.item_id]["count"]) for i in wl.items})
                orac = sq.place_with_beliefs(wl, {b: float(v) for b, v in fut.items()})
                span = heur - orac
                pub, dl = sq.attainable_ceiling(wl, vis), sq.attainable_ceiling(dead_last(wl, dead), vis)
                rows.append({"trace": name, "actions": space, "n_visible": len(vis), "n_dead_visible": len(dead),
                             "published_ceiling_pct": round(100 * (heur - pub) / span, 2),
                             "dead_last_ceiling_pct": round(100 * (heur - dl) / span, 2)})
                print(rows[-1], flush=True)
    finally:
        sq.LEGAL_ACTIONS = ALL
    (ROOT / "experiments" / "ceiling_ties.json").write_text(json.dumps(
        {"config": {"split": 0.5, "dead_visible": "visible at the split, no future access",
                    "dead_last": "dead visible blocks ordered after every other block (the belief-0 tie)",
                    "toys": "200 random workloads, 6 visible + 4 new uniform blocks, seed 0",
                    "exhaustive": "every feasible tier assignment of the visible blocks; new blocks by "
                                  "the default rule (fastest tier with room, creation order)"},
         "toys": toys, "rows": rows}, indent=2) + "\n")
    print("wrote experiments/ceiling_ties.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
