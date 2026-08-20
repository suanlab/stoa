#!/usr/bin/env python3
"""Sampling study: how much of a result is the sample?

A headline claim in an earlier draft — that a learned policy captures 66–88% of the
placement headroom on agentic traces — reversed to −48% when the conversation count grew
from 70 to 150. The cause was not the workload: `_iter_conversations` drew a *sorted
prefix*, so the two "independent samples" were nested, and file index correlates with
conversation length and reuse rate (mean requests per conversation falls 140 to 37 from
files 1–20 to 71–150). The claim was a curve read at two adjacent points of a parameter
we had fixed for convenience.

This script is the control that was missing. For each sample size it draws several
RANDOM conversation subsets (`seed=` in the loader), runs the same placement pipeline,
and reports mean ± 95% CI. A result that does not survive this table should not be
stated as a result.

CPU. Usage: python3 scripts/run_sampling_study.py [--sizes 20,40,80] [--draws 5]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_kvct_placement import (  # noqa: E402
    FEATURE_SETS, featurize, fit_regression, place_with_beliefs_at, predict_counts,
)
from train_session_features import auc, fit_logistic, predict  # noqa: E402

from stoa.kvct import load_kvct_with_meta  # noqa: E402


def one_draw(n_convs: int, seed: int, feature_set: str):
    """One random draw: fit on half the conversations, place on the disjoint half."""
    wl, meta = load_kvct_with_meta(limit=n_convs, seed=seed)
    split = wl.time_split(0.5)
    rows, ids, convs, y_bin, y_cnt = featurize(wl, meta, split)
    if not ids:
        return None
    uniq = sorted(set(convs))
    train = set(uniq[: len(uniq) // 2])
    tr = [i for i, c in enumerate(convs) if c in train]
    te = [i for i, c in enumerate(convs) if c not in train]
    if not tr or not te:
        return None

    keep = {ids[i] for i in te}
    wl.items = [i for i in wl.items if i.item_id in keep]
    wl.traces = {k: v for k, v in wl.traces.items() if k in keep}
    wl.accesses = [(t, b) for t, b in wl.accesses if b in keep]

    stats = wl.prefix_stats(split)
    fut = wl.future_counts(split)
    c_pref = place_with_beliefs_at(wl, {i.item_id: stats[i.item_id]["count"] for i in wl.items}, split)
    c_orac = place_with_beliefs_at(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items}, split)
    span = c_pref - c_orac
    if span <= 0:
        return None

    Xtr = [rows[feature_set][i] for i in tr]
    Xte = [rows[feature_set][i] for i in te]
    m = fit_logistic(Xtr, [y_bin[i] for i in tr], epochs=2000, lr=2.0)
    a = auc(predict(m, Xte), [y_bin[i] for i in te])
    rg = predict_counts(fit_regression(Xtr, [y_cnt[i] for i in tr]), Xte)
    c = place_with_beliefs_at(wl, {ids[i]: v for i, v in zip(te, rg)}, split)
    return {"auc": a, "captured": 100.0 * (c_pref - c) / span,
            "headroom": 100.0 * span / c_pref, "eval_blocks": len(te)}


# Student-t two-sided 97.5% quantiles, df = 1..9. Needed because the normal quantile 1.96
# is only correct when sigma is KNOWN; with sigma estimated from four draws the right
# multiplier is t(3) = 3.1824, and using 1.96 understates the half-width by 62%. An earlier
# version of this file did exactly that, and the resulting figure reached the abstract --
# the same small-sample error the paper's own protocol section warns about.
_T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
         6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622}


def spread(xs):
    """Return (mean, t-based half-width, min, max).

    The half-width is reported for continuity with earlier revisions, but at these sample
    sizes the OBSERVED RANGE is the honest summary: it assumes no distribution, and
    `captured` is a ratio whose denominator is itself random per draw, so a Gaussian
    interval on it is mis-specified as well as mis-quantiled.
    """
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0, xs[0], xs[0]
    t = _T975.get(len(xs) - 1, 1.96)
    return m, t * statistics.stdev(xs) / math.sqrt(len(xs)), min(xs), max(xs)


def ci95(xs):
    """Back-compatible shim: mean and t-based half-width."""
    m, half, _lo, _hi = spread(xs)
    return m, half


def main() -> None:
    p = argparse.ArgumentParser(description="Sampling study with confidence intervals")
    p.add_argument("--sizes", type=str, default="20,40,80")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--features", type=str, default="block+session+think",
                   choices=list(FEATURE_SETS))
    p.add_argument("--out", type=str, default="experiments/sampling_study.json")
    args = p.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    print(f"\nSampling study — {args.draws} random draws per size, features={args.features}")
    print(f"{'convs':>6} | {'headroom':>18} | {'AUC':>16} | {'headroom captured':>22}")
    print("-" * 74)
    rows = []
    for n in sizes:
        res = [r for s in range(args.draws) if (r := one_draw(n, 1000 + s, args.features))]
        if not res:
            continue
        hm, hh, _, _ = spread([r["headroom"] for r in res])
        am, ah, _, _ = spread([r["auc"] for r in res])
        cm, ch, clo, chi = spread([r["captured"] for r in res])
        print(f"{n:>6} | {hm:8.1f}% ± {hh:5.1f} | {am:7.3f} ± {ah:5.3f} | "
              f"{cm:+11.1f}% [obs {clo:+.0f}, {chi:+.0f}]")
        rows.append({"convs": n, "draws": len(res),
                     "headroom_mean": round(hm, 2), "headroom_ci95_t": round(hh, 2),
                     "auc_mean": round(am, 4), "auc_ci95_t": round(ah, 4),
                     "captured_mean": round(cm, 2), "captured_ci95_t": round(ch, 2),
                     # The honest summary at n=4: no distributional assumption.
                     "captured_observed_min": round(clo, 2),
                     "captured_observed_max": round(chi, 2),
                     "per_draw_captured": [round(r["captured"], 2) for r in res]})

    print(f"\nRead the spread, not the point. With {args.draws} draws a parametric interval")
    print("rests on a normality assumption the data cannot support, so the OBSERVED RANGE is")
    print("the summary to quote; the t-based half-width is kept only for continuity.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {"draws": args.draws, "features": args.features,
                                          "sampling": "random by seed, conversation-disjoint fit/eval"},
                               "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
