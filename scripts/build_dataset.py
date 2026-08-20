#!/usr/bin/env python3
"""Build the M0-3 offline dataset + oracle upper-bound baseline.

Deliverable for docs/research_plan.md §15 milestone M0-3:
generate a synthetic trace, compute the three offline reference points
(oracle upper bound / capacity-feasible greedy / naive), print a table, and
persist the report to experiments/ for reproducibility.

Usage:
    python3 scripts/build_dataset.py [--items N] [--horizon T] [--zipf S] [--seed K]

Stdlib-only; no install required (pythonpath=src is set in pyproject.toml, but we
also add src/ here so the script runs standalone).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.oracle import run  # noqa: E402
from stoa.traces import WorkloadConfig  # noqa: E402


def _fmt(result) -> str:
    top_tier = ", ".join(f"{k}:{v}" for k, v in sorted(result.tier_hist.items(), key=lambda x: -x[1]))
    return f"{result.name:<20} cost={result.total_cost:12.3f}   [{top_tier}]"


def main() -> None:
    p = argparse.ArgumentParser(description="STOA M0-3 offline dataset + oracle baseline")
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--horizon", type=int, default=1_000)
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="experiments/m0_oracle_baseline.json")
    args = p.parse_args()

    cfg = WorkloadConfig(n_items=args.items, horizon=args.horizon, zipf_s=args.zipf, seed=args.seed)
    report = run(cfg)

    print(f"\nSTOA M0-3 offline baseline  (items={args.items}, horizon={args.horizon}, "
          f"zipf={args.zipf}, seed={args.seed})")
    print("-" * 78)
    print(_fmt(report.oracle_upper_bound))
    print(_fmt(report.greedy_belady))
    print(_fmt(report.naive))
    print("-" * 78)
    print(f"greedy gap vs oracle ceiling : {report.greedy_gap_pct:6.2f}%")
    print(f"greedy savings vs naive      : {report.greedy_savings_vs_naive_pct:6.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
