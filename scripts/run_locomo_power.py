#!/usr/bin/env python3
"""How much LoCoMo data would the placement-vs-retrieval question have needed?

The paper reports that on LoCoMo the query-agnostic arms are not statistically separable,
and that two runs of ours rank them oppositely. Reporting a null result is only honest if it
comes with the sample size that would have resolved it -- otherwise "we could not tell"
reads as "there is nothing there", which is a different and unsupported claim.

This script answers three questions from the artifacts we already have, with no new LLM
calls:

  1. What are the exact two-sided p-values for each pairwise comparison, at each budget?
  2. How many scored questions per arm would 80% power require, at the observed effect size?
  3. How often would two runs of our sizes (16 and 30 questions) rank the arms oppositely
     purely by chance? If that probability is high, the reversal we observed is not evidence.

Everything is exact enumeration or explicit simulation (`stoa.stats`); scipy is unavailable
here and the samples are far too small for a normal approximation.

CPU, about a minute. Usage: python3 scripts/run_locomo_power.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.stats import fisher_exact, n_for_power, p_order_reversal  # noqa: E402

RUNS = {
    # name -> (artifact, questions scored per arm)
    "leak-free (3 dialogues x 10 questions)": ("eval_locomo_leakfree.json", 30),
    "companion (2 dialogues x 8 questions)": ("eval_locomo_baselines.json", 16),
}
PAIRS = (("stoa", "random"), ("retrieval", "stoa"), ("stoa", "centrality"))
POOL = 1986      # total LoCoMo questions across the 10 released dialogues


def main() -> None:
    ap = argparse.ArgumentParser(description="LoCoMo power analysis")
    ap.add_argument("--out", type=str, default="experiments/locomo_power.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp = Path(__file__).resolve().parent.parent / "experiments"
    report = {"config": {"pool_questions": POOL, "seed": args.seed,
                         "test": "two-sided Fisher exact; power by simulation"},
              "runs": {}, "power": [], "reversal": []}

    for label, (fname, n) in RUNS.items():
        p = exp / fname
        if not p.exists():
            print(f"skipping {label}: missing {fname}")
            continue
        d = json.loads(p.read_text())
        res, fracs = d["results"], d["budget_fractions"]
        print(f"\n=== {label}, n={n} scored questions per arm ===")
        print(f"{'budget':>7} | {'comparison':<22} | {'counts':>9} | {'p (Fisher)':>10}")
        print("-" * 60)
        rows = []
        for i, frac in enumerate(fracs):
            for hi, lo in PAIRS:
                if hi not in res or lo not in res:
                    continue
                a, c = round(res[hi][i] * n), round(res[lo][i] * n)
                pv = fisher_exact(a, n - a, c, n - c)
                print(f"{frac:7.0%} | {hi + ' vs ' + lo:<22} | {a:>3}/{n} {c:>3}/{n} | {pv:10.3f}")
                rows.append({"budget_frac": frac, "high": hi, "low": lo,
                             "hits_high": a, "hits_low": c, "n": n, "fisher_p": round(pv, 4)})
        # Nine comparisons per run (3 budgets x 3 pairs). Reporting a raw p of 0.01 from
        # nine tests without saying so would be exactly the kind of quiet over-claim this
        # paper is about, so the corrected threshold is printed alongside.
        thr = 0.05 / len(rows)
        surv = [r for r in rows if r["fisher_p"] < thr]
        print(f"  Bonferroni threshold for {len(rows)} comparisons: p < {thr:.4f}; "
              f"{len(surv)} survive"
              + (f" ({', '.join(r['high'] + ' vs ' + r['low'] + ' @ ' + format(r['budget_frac'], '.0%') for r in surv)})"
                 if surv else ""))
        report["runs"][label] = {"artifact": fname, "n_per_arm": n, "rows": rows,
                                 "bonferroni_threshold": round(thr, 5),
                                 "n_surviving_correction": len(surv)}

    # --- 2. required sample size, at the effect the leak-free run estimates -------------
    d = json.loads((exp / "eval_locomo_leakfree.json").read_text())
    res, fracs = d["results"], d["budget_fractions"]
    print("\n=== questions per arm for 80% power at the observed effect ===")
    print(f"{'budget':>7} | {'comparison':<22} | {'rates':>13} | {'n needed':>9} | {'vs 30 run':>9}")
    print("-" * 74)
    for i, frac in enumerate(fracs):
        for hi, lo in PAIRS:
            p1, p2 = res[hi][i], res[lo][i]
            need = n_for_power(p1, p2, target=0.80, seed=args.seed)
            factor = f"{need / 30:.0f}x" if need else "—"
            shown = f"{need}" if need else "> 4000"
            print(f"{frac:7.0%} | {hi + ' vs ' + lo:<22} | {p1:.3f} vs {p2:.3f} | "
                  f"{shown:>9} | {factor:>9}")
            report["power"].append({"budget_frac": frac, "high": hi, "low": lo,
                                    "rate_high": p1, "rate_low": p2,
                                    "n_per_arm_for_80pct_power": need})

    # --- 3. is the between-run reversal surprising? ------------------------------------
    print("\n=== P(two runs of n=16 and n=30 rank the arms oppositely, by chance alone) ===")
    for i, frac in enumerate(fracs):
        p1, p2 = res["stoa"][i], res["retrieval"][i]
        pr = p_order_reversal(p1, p2, 16, 30, seed=args.seed)
        print(f"{frac:7.0%} | stoa {p1:.3f} vs retrieval {p2:.3f} -> reversal p = {pr:.3f}")
        report["reversal"].append({"budget_frac": frac, "rate_stoa": p1,
                                   "rate_retrieval": p2, "p_reversal": round(pr, 4)})

    needed = [r["n_per_arm_for_80pct_power"] for r in report["power"]
              if r["n_per_arm_for_80pct_power"]]
    if needed:
        print(f"\nSmallest resolvable comparison needs {min(needed)} questions per arm; "
              f"we scored 30. LoCoMo holds {POOL}.")
        report["summary"] = {"min_n_needed": min(needed), "max_n_needed": max(needed),
                             "n_scored": 30, "pool": POOL,
                             "shortfall_factor": round(min(needed) / 30, 1)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
