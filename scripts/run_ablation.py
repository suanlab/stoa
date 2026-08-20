#!/usr/bin/env python3
"""M3-6 deliverable: per-axis (축별) ablation — the RQ1 basis.

Two complementary views on the same synthetic workload:

  * TIER axis (online caching, eval/online.py): Belady oracle vs LRU / H2O / static
    across hot-tier sizes. Shows the heuristic ordering and the oracle gap that a
    learned tier controller aims to close.
  * REPRESENTATION / JOINT axis (static placement, eval/oracle.py): oracle upper
    bound vs capacity-feasible greedy vs naive. Shows the token-cost savings from
    choosing representation, which the tier axis (fixed plaintext) cannot capture.

Writes a JSON report to experiments/ and prints a table. Stdlib-only.

Usage:
    python3 scripts/run_ablation.py [--items N] [--horizon T] [--zipf S] [--seed K]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.online import run_online  # noqa: E402
from stoa.eval.oracle import run as run_static  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402

CACHE_FRACS = (0.05, 0.10, 0.20)
SHOWN = ("belady", "lru", "h2o", "static")   # learned needs a fitted predictor (see train_controller.py)


def main() -> None:
    p = argparse.ArgumentParser(description="STOA M3-6 per-axis ablation (E1)")
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--horizon", type=int, default=8_000)
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="experiments/m3_ablation.json")
    args = p.parse_args()

    cfg = WorkloadConfig(n_items=args.items, horizon=args.horizon, zipf_s=args.zipf, seed=args.seed)
    wl = generate_workload(cfg)

    # --- TIER axis: online eviction across cache sizes ---
    tier: dict[str, dict[str, dict]] = {}
    print(f"\nTIER axis — online eviction hit-rate (items={args.items}, horizon={args.horizon}, "
          f"zipf={args.zipf})")
    print(f"{'cache_frac':>10} | " + " | ".join(f"{p:>8}" for p in SHOWN) + " |  oracle_gap")
    print("-" * 72)
    for frac in CACHE_FRACS:
        res = run_online(wl, cache_frac=frac)              # heuristics + oracle (no learned here)
        tier[f"{frac:.2f}"] = {p: res[p].to_dict() for p in res}
        rates = {p: res[p].hit_rate for p in res}
        # gap = how many more hits Belady gets than the best heuristic (pp)
        best_heur = max(rates["lru"], rates["h2o"], rates["static"])
        gap_pp = 100.0 * (rates["belady"] - best_heur)
        row = " | ".join(f"{rates[p]*100:7.2f}%" for p in SHOWN)
        print(f"{frac:>10.2f} | {row} | {gap_pp:+8.2f}pp")

    # --- REPRESENTATION / JOINT axis: static placement ---
    static = run_static(cfg)
    print(f"\nREPR/JOINT axis — static placement weighted cost")
    print("-" * 72)
    print(f"  oracle_upper_bound : {static.oracle_upper_bound.total_cost:10.3f}  (ceiling)")
    print(f"  greedy_belady      : {static.greedy_belady.total_cost:10.3f}  "
          f"(-{static.greedy_savings_vs_naive_pct:.1f}% vs naive)")
    print(f"  naive              : {static.naive.total_cost:10.3f}")

    report = {
        "config": {"n_items": args.items, "horizon": args.horizon, "zipf_s": args.zipf, "seed": args.seed},
        "tier_axis_online": tier,
        "repr_joint_axis_static": static.to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
