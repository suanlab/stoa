#!/usr/bin/env python3
"""M9-12 (torch-free): budget-conditioned control frontier — RQ3 controllability.

Sweeps the serve-time token budget B and shows one controller (eval/budget.py)
tracing the achievable (tokens, $, latency) frontier, versus budget-agnostic
static baselines (all-plaintext, all-vector, all-latent) that are single points.

Headline (RQ3): no single static policy is right across budgets — all-plaintext
violates any tight budget, all-latent overspends on loose ones — while the
budget-conditioned controller returns the min-$ feasible placement for every B.

Honest caveat (RQ1): under the placeholder cost model `vector` promotion is very
cheap, so uniform-vector is already near-efficient and the frontier's dollar
spread is small. Real-trace calibration (M0-3, pending) is expected to widen the
regime where budget-conditioning pays off.

Usage:  python3 scripts/run_budget_frontier.py [--items N] [--horizon T] [--zipf S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.eval.budget import frontier, static_point  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402

FRACS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.0)   # budget as a fraction of all-plaintext tokens


def main() -> None:
    p = argparse.ArgumentParser(description="STOA M9-12 budget-conditioned frontier (RQ3)")
    p.add_argument("--items", type=int, default=64)
    p.add_argument("--horizon", type=int, default=8_000)
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="experiments/m9_budget_frontier.json")
    args = p.parse_args()

    wl = generate_workload(WorkloadConfig(n_items=args.items, horizon=args.horizon,
                                          zipf_s=args.zipf, seed=args.seed))
    t0 = sum(wl.n_accesses(i.item_id) for i in wl.items) * 800   # all-plaintext tokens
    budgets = [int(f * t0) for f in FRACS]
    pts = frontier(wl, budgets)

    statics = {"all_plaintext": static_point(wl, 0), "all_vector": static_point(wl, 1),
               "all_latent": static_point(wl, 2)}

    print(f"\nM9-12 budget-conditioned frontier (items={args.items}, horizon={args.horizon}, "
          f"zipf={args.zipf}); all-plaintext tokens={t0:,}")
    print(f"{'budget B':>12} | {'tokens_used':>12} | {'dollars':>9} | {'promotions':>10} | feasible")
    print("-" * 72)
    for pt in pts:
        print(f"{pt.budget_tokens:>12,} | {pt.tokens_used:>12,} | {pt.dollar_cost:>9.4f} | "
              f"{pt.promotions:>10} | {pt.feasible}")

    print("\nstatic baselines (single points — right for only one budget):")
    for name, sp in statics.items():
        print(f"  {name:<14} tokens={sp.tokens_used:>10,}  ${sp.dollar_cost:.4f}")

    # RQ3 controllability: which budgets does each static baseline violate?
    tight = budgets[-2]   # 5% budget
    print(f"\nat a tight budget B={tight:,}:")
    print(f"  all_plaintext feasible? {statics['all_plaintext'].tokens_used <= tight}  "
          f"(tokens={statics['all_plaintext'].tokens_used:,})")
    print(f"  controller    feasible? {pts[-2].feasible}  "
          f"(tokens={pts[-2].tokens_used:,}, ${pts[-2].dollar_cost:.4f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"items": args.items, "horizon": args.horizon, "zipf": args.zipf, "seed": args.seed},
        "all_plaintext_tokens": t0,
        "frontier": [pt.to_dict() for pt in pts],
        "static_baselines": {k: v.to_dict() for k, v in statics.items()},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
