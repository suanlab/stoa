#!/usr/bin/env python3
"""Place blocks when they ARRIVE, not at a split instant. The decisive experiment.

Every placement result in this project so far fixes a split instant, builds features from
the prefix, and places the blocks that already exist. That protocol makes 97% of the
hindsight oracle's advantage unreachable by construction: 45-46% of blocks have no
pre-split access at all, and they carry the overwhelming majority of the gap. A policy that
cannot express an opinion about them cannot compete for their benefit, so a flat "captured"
number says more about the protocol than about learnability.

This script targets exactly that 97%. At each request, every block appearing for the FIRST
time is placed immediately, from request-level context available at that instant:

  * the shared-prefix session id (first block hash of the request) and how much that session
    has been seen so far -- its request count, rate and recency
  * the block's position within the request, and the request's length
  * wall-clock inter-arrival time for the session (Mooncake carries real timestamps)
  * how many distinct blocks have been seen so far -- a proxy for warm-up

No feature reads the future. The model is fitted on the FIRST HALF of the trace, in arrival
order, with labels supplied by what the second half turns out to contain; it then places
the second half's arrivals online. Cost is billed at the true future with the same weighted
cost function every other experiment uses, so the numbers are directly comparable.

Three arms, identical placement machinery:
  arrival-learned  logistic model over the features above, decided at arrival
  arrival-constant no model: every arrival gets the same belief (the do-nothing control)
  split-oracle     the clairvoyant belief restricted to split-visible blocks -- the ceiling
                   the previous protocol could reach

If the learned arm captures a meaningful share of the gap the split protocol could not
touch, the decoupling result is overturned. If it does not, the negative result becomes far
stronger, because it then holds in the regime where the headroom actually lives.

CPU, a few minutes per trace. Usage: python3 scripts/run_arrival_admission.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import (attainable_ceiling, place_with_beliefs,  # noqa: E402
                             prefix_greedy_cost, quantile_match)
from train_session_features import auc, fit_logistic, parse_requests, predict  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}


def arrival_features(path: str, max_requests: int):
    """One row per block, at the instant it first appears. No feature reads the future.

    Returns (X, y, ids, arrival_step) where `y` is the block's total future access count
    after arrival -- the quantity a placement policy would want to know.
    """
    reqs = parse_requests(path, max_requests)
    blk_steps: dict[str, list[int]] = defaultdict(list)
    for t, _sid, blocks in reqs:
        for b in blocks:
            blk_steps[b].append(t)

    sess_steps: dict[str, list[int]] = defaultdict(list)
    seen: set[str] = set()
    X, y, ids, arrival = [], [], [], []
    for t, sid, blocks in reqs:
        prev = sess_steps[sid]
        s_cnt = len(prev)
        s_rec = (t - prev[-1]) if prev else t
        s_age = (t - prev[0]) if prev else 0
        s_rate = s_cnt / max(s_age, 1)
        n_req = len(blocks)
        for pos, b in enumerate(blocks):
            if b in seen:
                continue
            seen.add(b)
            X.append([
                1.0,
                math.log1p(s_cnt),                 # how established this session is
                math.log1p(s_rec),                 # session recency
                math.log1p(s_age),                 # session age
                s_rate,                            # session request rate
                math.log1p(pos),                   # position within the request
                math.log1p(n_req),                 # request length
                pos / max(n_req, 1),               # relative position
                math.log1p(len(seen)),             # warm-up proxy
            ])
            # Label: accesses AFTER this arrival. Known only in hindsight; used for fitting
            # on the training half and for billing, never as an input feature.
            y.append(float(sum(1 for s in blk_steps[b] if s > t)))
            ids.append(b)
            arrival.append(t)
        sess_steps[sid].append(t)
    return X, y, ids, arrival


def main() -> None:
    ap = argparse.ArgumentParser(description="Arrival-time admission placement")
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--out", type=str, default="experiments/arrival_admission.json")
    args = ap.parse_args()
    n_req = args.requests or 10 ** 9

    rows = []
    print(f"\nArrival-time admission ({'full traces' if not args.requests else n_req})")
    print(f"{'trace':<13} | {'arm':<17} | {'AUC':>6} | {'of gap':>8} | {'of reach':>9}")
    print("-" * 66)

    for name, path in TRACES.items():
        if not Path(path).exists():
            sys.exit(f"missing {path} — see src/stoa/mooncake.py")
        wl = load_mooncake(path, max_requests=args.requests or None)
        X, y, ids, arrival = arrival_features(path, n_req)

        # Temporal split IN ARRIVAL ORDER: fit on blocks that arrived in the first half,
        # place the ones that arrive later. This is the deployable protocol -- no block is
        # scored by a model that saw its own future.
        cut = arrival[len(arrival) // 2] if arrival else 0
        tr_idx = [i for i, t in enumerate(arrival) if t <= cut]
        ev_idx = [i for i, t in enumerate(arrival) if t > cut]
        if len(tr_idx) < 100 or len(ev_idx) < 100:
            print(f"{name}: too few arrivals to split ({len(tr_idx)}/{len(ev_idx)})")
            continue

        # Binary target: will this block ever be used again after arrival?
        w = fit_logistic([X[i] for i in tr_idx], [1.0 if y[i] > 0 else 0.0 for i in tr_idx])
        scores = predict(w, [X[i] for i in ev_idx])
        a_auc = auc(scores, [1.0 if y[i] > 0 else 0.0 for i in ev_idx])

        ev_ids = [ids[i] for i in ev_idx]
        st = max(1, wl.time_split(0.5))
        fut = wl.future_counts(st)
        ref = [float(fut.get(i, 0)) for i in ev_ids]

        c_pref = prefix_greedy_cost(wl)
        c_orac = place_with_beliefs(
            wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
        c_ceil = attainable_ceiling(wl, ev_ids)
        span, reach = c_pref - c_orac, c_pref - c_ceil

        def score(vals):
            belief = dict(zip(ev_ids, quantile_match(list(vals), ref)))
            c = place_with_beliefs(wl, belief)
            return (100.0 * (c_pref - c) / span if span > 0 else 0.0,
                    100.0 * (c_pref - c) / reach if reach > 0 else 0.0)

        # The constant arm is NOT "no policy": quantile matching gives every block the same
        # magnitude, and the stable sort then places them in ARRIVAL order, so it is
        # first-come-first-served admission with a fixed action. The shuffled arm isolates
        # how much of its benefit comes from that ordering rather than from the fixed action.
        import random as _r
        _rng = _r.Random(17)
        shuffled = list(range(len(ev_idx)))
        _rng.shuffle(shuffled)
        arms = {"arrival-learned": (a_auc, scores),
                "arrival-fcfs": (0.5, [0.5] * len(ev_idx)),
                "arrival-shuffled": (0.5, [float(v) for v in shuffled])}
        for arm, (au, vals) in arms.items():
            g, r = score(vals)
            print(f"{name:<13} | {arm:<17} | {au:6.3f} | {g:7.1f}% | {r:8.1f}%")
            rows.append({"trace": name, "arm": arm, "auc": round(au, 4),
                         "captured_pct": round(g, 2),
                         "captured_of_attainable_pct": round(r, 2),
                         "n_arrivals_scored": len(ev_idx)})
        ceil_g = 100.0 * reach / span if span > 0 else 0.0
        print(f"{name:<13} | {'arrival-ceiling':<17} | {1.0:6.3f} | {ceil_g:7.1f}% | {100.0:8.1f}%")
        rows.append({"trace": name, "arm": "arrival-ceiling", "auc": 1.0,
                     "captured_pct": round(ceil_g, 2),
                     "captured_of_attainable_pct": 100.0,
                     "n_arrivals_scored": len(ev_idx)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"requests": args.requests or "full",
                    "protocol": ("blocks placed at first appearance from request-level "
                                 "context; temporal split in arrival order; beliefs "
                                 "quantile-matched to a common magnitude scale"),
                    "why": ("the split-instant protocol leaves 97% of the hindsight gap "
                            "unreachable; this targets that 97%")},
         "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
