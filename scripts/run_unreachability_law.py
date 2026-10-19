#!/usr/bin/env python3
"""H5: is the split-instant unreachable share a law of the trace's birth and reuse process?

Measured U(s) comes from reachable_ceiling.json (splits 0.3..0.7). Two predictions of it:
  A(s)   the share of post-split accesses that land on blocks born at or after s (empirical);
  Ahat(s) the same share PREDICTED from two curves only: the birth rate (births per step, from the
          whole trace) and the mean cumulative reuse function G(u) = expected accesses within age u
          of a block's birth, estimated with right-censoring handled (a block born at t contributes
          to G(u) only if t + u is inside the trace), as in a Nelson-Aalen mean cumulative function.
  Ahat(s) = sum_{t>=s} b(t) G(T-t) / [ sum_{t>=s} b(t) G(T-t) + sum_{t<s} b(t) (G(T-t) - G(s-t)) ]
H5 predicts |U - Ahat| <= 3 points at every split on every trace (falsified if > 5 anywhere).
Usage: python3 scripts/run_unreachability_law.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arrival_admission import TRACES  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402


def main() -> int:
    rc = json.loads((ROOT / "experiments" / "reachable_ceiling.json").read_text())
    rows = []
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        lo, hi = wl.accesses[0][0], wl.accesses[-1][0]
        T = hi + 1
        births = [0] * T
        acc_at_age = [0] * T                # accesses at age a (birth access at a = 0)
        for tr in wl.traces.values():
            s = tr.access_steps
            births[s[0]] += 1
            for x in s[1:]:                 # the creation access is a compulsory miss (notebook AS)
                acc_at_age[x - s[0]] += 1
        # censoring: blocks observable at age a = those born at or before T-1-a
        cum_births = [0] * T
        run = 0
        for t in range(T):
            run += births[t]
            cum_births[t] = run
        G, g_run = [0.0] * T, 0.0
        for a in range(T):
            at_risk = cum_births[T - 1 - a]
            g_run += acc_at_age[a] / at_risk if at_risk else 0.0
            G[a] = g_run
        for r in rc["traces"][name]["rows"]:
            frac = r["split_frac"]
            s = max(1, wl.time_split(frac))
            new = sum(births[t] * G[T - 1 - t] for t in range(s, T))
            old = sum(births[t] * (G[T - 1 - t] - (G[s - 1 - t] if s - 1 - t >= 0 else 0.0))
                      for t in range(lo, s))
            a_hat = 100.0 * new / (new + old)
            first = {b: tr.access_steps[0] for b, tr in wl.traces.items()}
            # billable post-split accesses only: a block's creation access is not billed
            post = [(st_, b) for st_, b in wl.accesses if st_ >= s and st_ != first[b]]
            a_emp = 100.0 * sum(1 for _, b in post if first[b] >= s) / len(post)
            rows.append({"trace": name, "split_frac": frac, "U_measured": r["unreachable_pct"],
                         "A_empirical": round(a_emp, 2), "A_hat_model": round(a_hat, 2),
                         "err_model": round(a_hat - r["unreachable_pct"], 2),
                         "err_empirical": round(a_emp - r["unreachable_pct"], 2)})
            print(rows[-1])
    out = {"config": {"G": "censoring-corrected mean cumulative REUSES by age (creation excluded)",
                      "births": "whole trace"}, "rows": rows,
           "max_abs_err_model": max(abs(r["err_model"]) for r in rows),
           "max_abs_err_empirical": max(abs(r["err_empirical"]) for r in rows)}
    (ROOT / "experiments" / "unreachability_law.json").write_text(json.dumps(out, indent=2) + "\n")
    print("max |err| model", out["max_abs_err_model"], "empirical", out["max_abs_err_empirical"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
