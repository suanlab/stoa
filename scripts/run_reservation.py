#!/usr/bin/env python3
"""H5: how much of the split-time reference's gap is reservation? (§BH; ADVISE H5)

§AZ found that an exhaustive search over placements of the visible blocks beats the dead-last reference by
reserving fast slots for blocks that arrive later -- which needs the arrivals' futures. Newsvendor logic
says a reservation needs only their distribution: a fast slot should go to a visible block only if its
future count exceeds what an arriving block is expected to bring. This reference knows the visible
blocks' futures (as the others do) and the MEAN future count of the new blocks (not which block brings
it), and parks every visible block below that mean behind the new blocks:

  newsvendor   visible blocks with future count < mean future count of new blocks -> behind new blocks

Toys: the 200 random ten-block workloads of run_ceiling_ties.py (same seed and generator), tier-only and
all actions; recovered share = sum(dead-last - newsvendor) / sum(dead-last - exhaustive) over the toys.
Traces: the reference on the production traces and the synthetic control, split 0.5, tier-only.

Pre-registered (ADVISE H5): confirmed if newsvendor recovers at least half of the dead-last-to-exhaustive
gap on the toys (tier-only); rejected if it recovers under a quarter.
Usage: python3 scripts/run_reservation.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_ceiling_ties import SPACES, dead_last, exhaustive, split_sets, toy  # noqa: E402
from run_evidence_ladder import TIER, TRACES, ordered  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402


def newsvendor(wl, split=0.5):
    """(workload ordered for the reference, belief) -- visible futures, below-mean visible blocks last."""
    st, fut, vis, _ = split_sets(wl) if split == 0.5 else (None, None, None, None)
    vset = set(vis)
    new = [i.item_id for i in wl.items if i.item_id not in vset]
    mu = sum(fut.get(b, 0) for b in new) / len(new) if new else 0.0
    low = {b for b in vis if fut.get(b, 0) < mu}
    return ordered(wl, low), {b: (0.0 if b in low else float(fut.get(b, 0))) for b in vis}, mu


def main() -> int:
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    ALL = list(sq.LEGAL_ACTIONS)
    out = {"config": {"toys": "run_ceiling_ties.toy, 200, seed 0", "split": 0.5,
                      "registered": "confirmed if recovered >= 0.5 (toys, tier-only); rejected if < 0.25"},
           "toys": {}, "traces": []}
    try:
        for space, acts in SPACES.items():
            sq.LEGAL_ACTIONS = acts
            rng = random.Random(0)
            num = den = 0.0
            worse = 0
            for _ in range(200):
                wl = toy(rng)
                _, _, vis, dead = split_sets(wl)
                dl = sq.attainable_ceiling(dead_last(wl, dead), vis)
                ex = exhaustive(wl, acts, sim, w)
                w2, bel, _mu = newsvendor(wl)
                nv = sq.place_with_beliefs(w2, bel)
                num += dl - nv
                den += dl - ex
                worse += nv > dl + 1e-9
            out["toys"][space] = {"recovered_share": round(num / den, 4) if den else None,
                                  "newsvendor_worse_than_dead_last": worse, "n": 200}
            print(space, out["toys"][space], flush=True)
        sq.LEGAL_ACTIONS = TIER
        for name, path in TRACES.items():
            wl = load_mooncake(str(ROOT / path))
            st, fut, vis, dead = split_sets(wl)
            pre = wl.prefix_stats(st)
            heur = sq.place_with_beliefs(wl, {b: float(pre[b]["count"]) for b in vis})
            orac = sq.place_with_beliefs(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
            dl = sq.attainable_ceiling(dead_last(wl, dead), vis)
            w2, bel, mu = newsvendor(wl)
            nv = sq.place_with_beliefs(w2, bel)
            span = heur - orac
            row = {"trace": name, "mean_future_new": round(mu, 4),
                   "dead_last_pct": round(100 * (heur - dl) / span, 2),
                   "newsvendor_pct": round(100 * (heur - nv) / span, 2),
                   "parked_live_visible": sum(1 for b in vis if 0 < fut.get(b, 0) < mu)}
            out["traces"].append(row)
            print(row, flush=True)
    finally:
        sq.LEGAL_ACTIONS = ALL
    (ROOT / "experiments" / "reservation.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/reservation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
