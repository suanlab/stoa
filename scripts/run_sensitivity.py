#!/usr/bin/env python3
"""Cost-model sensitivity: how much of the headroom claim survives perturbed cost tables?

Our simulator's latency/token/promotion tables are illustrative placeholders (the
calibration gate against a real vLLM+LMCache deployment has not been passed), so every
number expressed as a weighted placement cost -- including "headroom" -- inherits that
uncertainty (docs/claims_dependency.md §B). Rather than asking readers to take the
magnitude on faith, we perturb the tables over several orders of magnitude and report
how the headroom moves.

The conclusions that matter are stated elsewhere in cost-independent terms (reuse
distributions, AUC, hit rate); this script bounds what the cost-dependent framing adds.

CPU, seconds. Usage: python3 scripts/run_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import stoa.simulator as sim_mod  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import reference_costs  # noqa: E402

# (promotion-cost multiplier, per-tier-latency multiplier, token-cost multiplier)
GRID = [
    ("baseline", 1.0, 1.0, 1.0),
    ("promotion 100x", 100.0, 1.0, 1.0),
    ("promotion 1000x", 1000.0, 1.0, 1.0),
    ("latency 10x", 1.0, 10.0, 1.0),
    ("latency 0.1x", 1.0, 0.1, 1.0),
    ("tokens 10x", 1.0, 1.0, 10.0),
    ("tokens 0.1x", 1.0, 1.0, 0.1),
    ("promotion 100x + tokens 10x", 100.0, 1.0, 10.0),
]


def main() -> None:
    path = "data/mooncake_toolagent_trace.jsonl"
    if not Path(path).exists():
        sys.exit(f"missing {path} — see src/stoa/mooncake.py for the download command")
    wl = load_mooncake(path, max_requests=2000)

    base_promote = dict(sim_mod._PROMOTE_COST)
    base_lat = dict(sim_mod._TIER_READ_MS)
    base_tok = dict(sim_mod._REPR_TOKENS)

    print("\nCost-model sensitivity (Mooncake toolagent, 2k requests)")
    print(f"{'perturbation':>28} | {'prefix-greedy':>13} | {'oracle':>10} | headroom")
    print("-" * 72)
    rows = []
    for name, pm, lm, tm in GRID:
        for k in base_promote:
            sim_mod._PROMOTE_COST[k] = base_promote[k] * pm
        for k in base_lat:
            sim_mod._TIER_READ_MS[k] = base_lat[k] * lm
        for k in base_tok:
            sim_mod._REPR_TOKENS[k] = int(base_tok[k] * tm)
        r = reference_costs(wl)
        print(f"{name:>28} | {r['prefix_greedy']:13.1f} | {r['oracle_future']:10.1f} | "
              f"{r['headroom_pct']:6.1f}%")
        rows.append({"perturbation": name, "promote_mult": pm, "latency_mult": lm,
                     "token_mult": tm, "prefix_greedy": round(r["prefix_greedy"], 2),
                     "oracle": round(r["oracle_future"], 2),
                     "headroom_pct": round(r["headroom_pct"], 2)})

    for k in base_promote:
        sim_mod._PROMOTE_COST[k] = base_promote[k]
    for k in base_lat:
        sim_mod._TIER_READ_MS[k] = base_lat[k]
    for k in base_tok:
        sim_mod._REPR_TOKENS[k] = base_tok[k]

    hs = [r["headroom_pct"] for r in rows]
    print(f"\nheadroom across the grid: min {min(hs):.1f}%  max {max(hs):.1f}%  "
          f"(baseline {rows[0]['headroom_pct']:.1f}%)")
    out = Path("experiments/sensitivity.json")
    out.write_text(json.dumps({"trace": "mooncake_toolagent", "requests": 2000,
                               "rows": rows,
                               "headroom_min": min(hs), "headroom_max": max(hs)}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
