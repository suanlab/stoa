#!/usr/bin/env python3
"""Sweep the last free parameter in the decoupling result: where the split instant falls.

Every Mooncake placement number in the paper is computed at `time_split(0.5)` -- half the
trace observed, half predicted. Nothing chose 0.5 except convenience, and the paper's own
protocol section says that reading a curve from one value of a parameter fixed for
convenience is how three of our defects happened. Trace length has since been swept to
exhaustion and model capacity has been swept across three model classes; the split instant
is what remains.

This is a SENSITIVITY band, not a confidence interval. The full traces are used and every
policy is deterministic, so there is no sampling distribution to draw from -- re-running
reproduces exactly. What varies is the question being asked: at split 0.3 the model sees
less history and must forecast further; at 0.7 the reverse.

The split instant is passed to BOTH feature construction and cost billing from a single
variable. Deriving them separately is precisely the reference-frame error that swung an
earlier result by 44 points (docs/claims_dependency.md).

CPU, a few minutes. Usage: python3 scripts/run_split_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stoa.gbdt import GBDT  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402
from train_session_features import TRACES, auc, build_features, fit_logistic, predict  # noqa: E402

SPLITS = (0.3, 0.4, 0.5, 0.6, 0.7)


def main() -> None:
    ap = argparse.ArgumentParser(description="Split-instant sensitivity")
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--out", type=str, default="experiments/split_sensitivity.json")
    args = ap.parse_args()
    n_req = args.requests or 10 ** 9

    wls = {}
    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path} — see src/stoa/mooncake.py for the download command")
        wls[name] = load_mooncake(path, max_requests=args.requests or None)

    rows = []
    print(f"\nSplit-instant sensitivity ({'full traces' if not args.requests else n_req})")
    print(f"{'split':>6} | {'trace':<13} | {'AUC lin':>7} | {'cap lin':>8} | "
          f"{'AUC gbdt':>8} | {'cap gbdt':>8} | {'gap':>6}")
    print("-" * 76)
    for frac in SPLITS:
        # One variable feeds features AND billing; see the module docstring.
        feats = {n: build_features(p, n_req, split_frac=frac) for n, p in TRACES.items()}
        for ev in TRACES:
            tr = next(n for n in TRACES if n != ev)
            Xe, _, ye, ids = feats[ev]
            Xt, _, yt, _ = feats[tr]
            wl = wls[ev]
            st = max(1, wl.time_split(frac))
            stats, fut = wl.prefix_stats(st), wl.future_counts(st)
            reused = [v for v in fut.values() if v > 0]
            mean_reuse = sum(reused) / max(1, len(reused))
            c_pref = place_with_beliefs(
                wl, {i.item_id: stats[i.item_id]["count"] for i in wl.items}, split_frac=frac)
            c_orac = place_with_beliefs(
                wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items}, split_frac=frac)
            span = c_pref - c_orac

            out = {}
            for tag, scores in (("lin", predict(fit_logistic(Xt, yt), Xe)),
                                ("gbdt", [float(v) for v in
                                          GBDT(n_trees=200, lr=0.1, max_depth=4)
                                          .fit(Xt, yt).predict(Xe)])):
                lo, hi = min(scores), max(scores)
                norm = [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in scores]
                c = place_with_beliefs(
                    wl, {i: p * mean_reuse for i, p in zip(ids, norm)}, split_frac=frac)
                out[tag] = (auc(scores, ye), 100.0 * (c_pref - c) / span if span > 0 else 0.0)
            gap = 100.0 * span / c_pref if c_pref > 0 else 0.0
            print(f"{frac:6.1f} | {ev:<13} | {out['lin'][0]:7.3f} | {out['lin'][1]:7.1f}% | "
                  f"{out['gbdt'][0]:8.3f} | {out['gbdt'][1]:7.1f}% | {gap:5.1f}%")
            rows.append({"split_frac": frac, "eval": ev, "train": tr,
                         "auc_linear": round(out["lin"][0], 4),
                         "captured_linear_pct": round(out["lin"][1], 2),
                         "auc_gbdt": round(out["gbdt"][0], 4),
                         "captured_gbdt_pct": round(out["gbdt"][1], 2),
                         "oracle_gap_pct": round(gap, 2)})

    caps = [r["captured_linear_pct"] for r in rows] + [r["captured_gbdt_pct"] for r in rows]
    gaps = [r["oracle_gap_pct"] for r in rows]
    print(f"\ncaptured across all splits, both traces, both models: "
          f"[{min(caps):+.1f}%, {max(caps):+.1f}%]")
    print(f"oracle gap across all splits: [{min(gaps):.1f}%, {max(gaps):.1f}%]")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(
        {"config": {"requests": args.requests or "full", "splits": list(SPLITS),
                    "note": "sensitivity band over a free parameter, NOT a sampling CI"},
         "summary": {"captured_min": round(min(caps), 2), "captured_max": round(max(caps), 2),
                     "oracle_gap_min": round(min(gaps), 2),
                     "oracle_gap_max": round(max(gaps), 2)},
         "results": rows}, indent=2))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
