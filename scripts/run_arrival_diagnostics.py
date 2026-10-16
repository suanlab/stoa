#!/usr/bin/env python3
"""Diagnostics behind section 5.4's two cautionary readings, as an artifact rather than a probe.

1. **Exposure time.** Within a finite trace, a block that arrives early in the evaluation period has
   more time to be reused than one that arrives late. Reported as the mean future access count of
   evaluation blocks arriving in the first vs. the last quarter of the evaluation period, and as the
   AUC with which "arrived earlier" alone predicts reuse -- the advantage first-come-first-served
   inherits from the evaluation window itself.
2. **The training-label leak of the earlier design.** That version labelled each training block with
   every later access and cut train/eval at the median block arrival, a different instant from the
   billing split. Reported: the share of training-label mass that fell after its own cut, and the gap
   between that cut and the billing split.

These numbers were first computed by an ad-hoc probe during the audit (AUDIT.md); the paper may only
state numbers that an artifact carries (docs/claims_dependency.md section AN), so they live here.

    python3 scripts/run_arrival_diagnostics.py
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_arrival_admission import TRACES, arrival_features  # noqa: E402
from stoa.causal import split_by_arrival  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402


def _auc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = sorted(s for s, l in zip(score, label) if not l)
    if not pos or not neg:
        return None
    tot = 0.0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def main() -> int:
    rows = []
    for name, path in TRACES.items():
        p = ROOT / path
        if not p.exists():
            sys.exit(f"missing {path} — see data/README.md")
        wl = load_mooncake(str(p))
        X, y, ids, arrival = arrival_features(str(p), 10 ** 9)
        st = max(1, wl.time_split(0.5))
        _, ev = split_by_arrival(arrival, st)
        end = max(arrival) + 1
        q = (end - st) // 4
        first = [y[i] for i in ev if arrival[i] < st + q]
        last = [y[i] for i in ev if arrival[i] >= end - q]
        mean_first, mean_last = sum(first) / len(first), sum(last) / len(last)
        auc_order = _auc([-arrival[i] for i in ev], [y[i] > 0 for i in ev])

        # The earlier design, reconstructed only to measure its leak.
        old_cut = arrival[len(arrival) // 2]
        steps = {b: wl.traces[b].access_steps for b in ids}
        tr_old = [i for i, t in enumerate(arrival) if t <= old_cut]
        total = sum(y[i] for i in tr_old)
        after = sum(sum(1 for s in steps[ids[i]] if s > old_cut) for i in tr_old)
        rows.append({"trace": name,
                     "eval_first_quarter_mean_future": round(mean_first, 3),
                     "eval_last_quarter_mean_future": round(mean_last, 3),
                     "first_over_last": round(mean_first / mean_last, 2),
                     "auc_arrival_order": round(auc_order, 3),
                     "old_design_cut": old_cut, "billing_split": st,
                     "old_design_label_mass_after_cut_pct": round(100.0 * after / total, 1)})
        r = rows[-1]
        print(f"{name:<13} early/late reuse {r['eval_first_quarter_mean_future']:.2f}/"
              f"{r['eval_last_quarter_mean_future']:.2f} (x{r['first_over_last']})  "
              f"AUC(order) {r['auc_arrival_order']:.3f}  old leak {r['old_design_label_mass_after_cut_pct']}%"
              f"  cut {old_cut} vs split {st}")
    out = {"config": {"split": "time_split(0.5)", "quarters": "of the evaluation period"},
           "rows": rows}
    (ROOT / "experiments" / "arrival_diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/arrival_diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
