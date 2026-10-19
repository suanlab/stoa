#!/usr/bin/env python3
"""H3: is arrival order's predictive power an artifact of unequal exposure?

In a finite window a block born early has more time to be reused than one born late, so "arrived
earlier" predicts reuse (AUC 0.55-0.58, arrival_diagnostics.json). Here every evaluated block gets
the SAME post-arrival horizon H: only blocks born in [split, end - H] are scored, each labelled by its
accesses in (birth, birth + H]. If the order signal is exposure, its AUC falls to ~0.5 and the
early/late reuse ratio to ~1; if it survives, it reflects session dynamics, not the window.

The learned ranker is refitted exactly as in run_arrival_admission.py (windowed pre-split labels) and
scored against the same equal-horizon labels. H = half the evaluation period.

CPU, a few minutes. Usage: python3 scripts/run_equal_horizon.py
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
from run_arrival_diagnostics import _auc  # noqa: E402
from stoa.causal import split_by_arrival, windowed_training_labels  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from train_session_features import fit_logistic, predict  # noqa: E402


def main() -> int:
    rows = []
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        X, y, ids, arrival = arrival_features(str(ROOT / path), 10 ** 9)
        st = max(1, wl.time_split(0.5))
        _, ev = split_by_arrival(arrival, st)
        end = max(arrival) + 1
        H = (end - st) // 2
        steps = {b: wl.traces[b].access_steps for b in ids}
        fit_idx, fit_lab = windowed_training_labels(arrival, ids, steps, st, st // 2)
        w = fit_logistic([X[i] for i in fit_idx], [1.0 if v > 0 else 0.0 for v in fit_lab])
        scored = [i for i in ev if arrival[i] + H < end]
        lab = []
        for i in scored:
            s = steps[ids[i]]
            a = arrival[i]
            lab.append(bisect.bisect_right(s, a + H) - bisect.bisect_right(s, a))
        pos = [v > 0 for v in lab]
        sc = predict(w, [X[i] for i in scored])
        q = max(1, (end - H - st) // 4)
        first = [v for i, v in zip(scored, lab) if arrival[i] < st + q]
        last = [v for i, v in zip(scored, lab) if arrival[i] >= end - H - q]
        window_pos = [y[i] > 0 for i in scored]   # the finite-window label, same blocks
        rows.append({
            "trace": name, "horizon_steps": H, "scored_blocks": len(scored),
            "auc_order_equal_horizon": round(_auc([-arrival[i] for i in scored], pos), 3),
            "auc_order_finite_window": round(_auc([-arrival[i] for i in scored], window_pos), 3),
            "auc_learned_equal_horizon": round(_auc(sc, pos), 3),
            "auc_learned_finite_window": round(_auc(sc, window_pos), 3),
            "first_over_last_equal_horizon": round((sum(first) / len(first)) / (sum(last) / len(last)), 2),
            "reuse_rate_equal_horizon": round(sum(pos) / len(pos), 3)})
        print(rows[-1])
    out = {"config": {"split": "time_split(0.5)", "horizon": "half the evaluation period",
                      "scored": "blocks born in [split, end - H]"}, "rows": rows}
    (ROOT / "experiments" / "equal_horizon.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/equal_horizon.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
