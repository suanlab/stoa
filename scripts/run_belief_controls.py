#!/usr/bin/env python3
"""The two controls Section 5.2 cited but never ran.

An earlier draft asserted that "rank-normalizing every belief changes the captured headroom
by about two points" and that "substituting true future counts recovers 95% of the interval".
Neither number came from anything in this repository: no script implemented rank
normalization, and no artifact contained a 95% figure. Both are computed here, for real, on
the corrected metric.

The corrected metric changes what these controls mean, so both are restated:

**Magnitude control.** `place_with_beliefs` is not ordering-invariant -- the per-item argmin
has thresholds in the belief, so two policies with identical rankings but different scales
place differently. The control is therefore: score each rung at its NATIVE magnitude and
again after `quantile_match` onto one common multiset. The gap between the two is exactly
what magnitude contributes, and it is the number the draft guessed at.

**Ceiling control.** The old "true future counts" arm was the oracle by construction, so it
could only ever return 100%. The meaningful version restricts clairvoyance to the blocks a
causal policy could see -- `attainable_ceiling` -- which is the correct denominator and the
subject of the retraction in docs/claims_dependency.md T.

CPU, a few minutes per trace. Usage: python3 scripts/run_belief_controls.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import (attainable_ceiling, place_with_beliefs,  # noqa: E402
                             prefix_greedy_cost, quantile_match)
from train_session_features import TRACES, build_features, fit_logistic, predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Belief magnitude and ceiling controls")
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/belief_controls.json")
    args = ap.parse_args()
    n_req = args.requests or 10 ** 9

    rows = []
    print(f"\nBelief controls ({'full traces' if not args.requests else n_req})")
    print(f"{'trace':<13} | {'rung':<10} | {'native':>9} | {'common scale':>13} | {'magnitude':>10}")
    print("  'magnitude' = native minus common-scale: what the SCALE contributes, not the order")
    print("-" * 74)

    feats = {n: build_features(p, n_req) for n, p in TRACES.items()}
    for ev, path in TRACES.items():
        tr = next(n for n in TRACES if n != ev)
        Xe, _, ye, ids = feats[ev]
        Xt, _, yt, _ = feats[tr]
        wl = load_mooncake(path, max_requests=args.requests or None)
        st = max(1, wl.time_split(0.5))
        fut = wl.future_counts(st)
        c_pref = prefix_greedy_cost(wl)
        c_orac = place_with_beliefs(
            wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
        c_ceil = attainable_ceiling(wl, ids)
        span, reach = c_pref - c_orac, c_pref - c_ceil
        ref = [float(fut.get(i, 0)) for i in ids]
        reused = [v for v in ref if v > 0]
        mean_reuse = sum(reused) / max(1, len(reused))

        rng = random.Random(args.seed + 991)
        arms = {
            "linear": predict(fit_logistic(Xt, yt), Xe),
            "count": [x[1] for x in Xe],
            "random": [rng.random() for _ in ye],
        }

        def cap(belief_by_id):
            c = place_with_beliefs(wl, belief_by_id)
            return (100.0 * (c_pref - c) / span if span > 0 else 0.0,
                    100.0 * (c_pref - c) / reach if reach > 0 else 0.0)

        for rung, s in arms.items():
            # Native: the scale each rung naturally produces, mapped to expected reuse the
            # way the original scripts did.
            lo, hi = min(s), max(s)
            nat = [(v - lo) / (hi - lo) * mean_reuse if hi > lo else 0.0 for v in s]
            n_full, n_reach = cap(dict(zip(ids, nat)))
            # Common scale: identical ordering, magnitudes replaced by the reference multiset.
            m_full, m_reach = cap(dict(zip(ids, quantile_match(list(s), ref))))
            print(f"{ev:<13} | {rung:<10} | {n_reach:8.1f}% | {m_reach:12.1f}% | "
                  f"{n_reach - m_reach:+9.1f}")
            rows.append({"trace": ev, "rung": rung,
                         "native_of_full_gap": round(n_full, 2),
                         "native_of_attainable": round(n_reach, 2),
                         "common_scale_of_full_gap": round(m_full, 2),
                         "common_scale_of_attainable": round(m_reach, 2),
                         "magnitude_contribution": round(n_reach - m_reach, 2)})

        ceil_full = 100.0 * reach / span if span > 0 else 0.0
        print(f"{ev:<13} | {'ceiling':<10} | {100.0:8.1f}% | {100.0:12.1f}% | "
              f"{0.0:+9.1f}   (= {ceil_full:.1f}% of the full hindsight gap)")
        rows.append({"trace": ev, "rung": "ceiling",
                     "native_of_full_gap": round(ceil_full, 2),
                     "native_of_attainable": 100.0,
                     "common_scale_of_full_gap": round(ceil_full, 2),
                     "common_scale_of_attainable": 100.0,
                     "magnitude_contribution": 0.0})

    mags = [abs(r["magnitude_contribution"]) for r in rows if r["rung"] != "ceiling"]
    print(f"\nmagnitude contributes between {min(mags):.1f} and {max(mags):.1f} points "
          f"of attainable headroom, depending on the rung and trace.")
    print("The draft asserted 'about two points' without running this.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"requests": args.requests or "full", "seed": args.seed,
                    "note": ("magnitude control = native scale minus common scale, same "
                             "ordering; ceiling control = clairvoyance restricted to blocks "
                             "a causal policy can see")},
         "summary": {"magnitude_min_points": round(min(mags), 2),
                     "magnitude_max_points": round(max(mags), 2)},
         "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
