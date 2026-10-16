#!/usr/bin/env python3
"""How much does the headline depend on the cost weights? (audit M4)

The placement cost model combines latency, money and context tokens with weights
`DEFAULT_WEIGHTS` = (lambda_latency 0.1 per ms, lambda_cost 1.0 per USD, lambda_tokens 0.0005 per
token), documented in `stoa.eval.oracle` as illustrative unit-normalising values and unchanged since
the artifact's first commit. The paper used to give the lambdas only as symbols. This sweeps each
non-numeraire weight an order of magnitude either way and reports the unreachable share -- the
paper's headline -- at the midpoint split, through the same routine as run_reachable_ceiling.py.

    python3 scripts/run_weight_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

# Forked workers must not inherit a live intra-op thread pool (see run_calibration_sensitivity).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_reachable_ceiling import TRACES, one  # noqa: E402  (same routine, same frame)
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import DEFAULT_WEIGHTS  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

VARIANTS = {
    "default": {},
    "lambda_latency x0.1": {"lambda_latency": DEFAULT_WEIGHTS.lambda_latency * 0.1},
    "lambda_latency x10": {"lambda_latency": DEFAULT_WEIGHTS.lambda_latency * 10},
    "lambda_tokens x0.1": {"lambda_tokens": DEFAULT_WEIGHTS.lambda_tokens * 0.1},
    "lambda_tokens x10": {"lambda_tokens": DEFAULT_WEIGHTS.lambda_tokens * 10},
}
PRODUCTION = ("conversation", "toolagent")


def _job(job):
    trace, variant = job
    wl = load_mooncake(str(ROOT / TRACES[trace]))
    w = replace(DEFAULT_WEIGHTS, **VARIANTS[variant])
    r = one(wl, 0.5, TieringSimulator(), w)
    return trace, variant, r


def main() -> int:
    jobs = [(t, v) for t in PRODUCTION for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        results = list(ex.map(_job, jobs))
    rows = []
    print(f"{'trace':<13} {'variant':<22} {'unreachable':>12}")
    for trace, variant, r in results:
        rows.append({"trace": trace, "variant": variant, "weights": {
            k: getattr(replace(DEFAULT_WEIGHTS, **VARIANTS[variant]), k)
            for k in ("lambda_latency", "lambda_cost", "lambda_tokens")},
            "unreachable_pct": r["unreachable_pct"], "full_hindsight_gap": r["full_hindsight_gap"]})
        print(f"{trace:<13} {variant:<22} {r['unreachable_pct']:>11.1f}%")
    u = [r["unreachable_pct"] for r in rows if r["unreachable_pct"] is not None]
    out = {"config": {"split_frac": 0.5, "default_weights": {
               "lambda_latency": DEFAULT_WEIGHTS.lambda_latency,
               "lambda_cost": DEFAULT_WEIGHTS.lambda_cost,
               "lambda_tokens": DEFAULT_WEIGHTS.lambda_tokens},
               "variants": list(VARIANTS)},
           "summary": {"unreachable_min": min(u), "unreachable_max": max(u)},
           "rows": rows}
    (ROOT / "experiments" / "weight_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nunreachable share across weight variants: {min(u):.1f}%-{max(u):.1f}%")
    print("wrote experiments/weight_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
