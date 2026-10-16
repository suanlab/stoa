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
from stoa.causal import (analogue_magnitudes, split_by_arrival,  # noqa: E402
                         windowed_training_labels)
from stoa.sequential import (attainable_ceiling, place_with_beliefs,  # noqa: E402
                             prefix_greedy_cost, quantile_match)
from train_session_features import auc, fit_logistic, parse_requests, predict  # noqa: E402

# Two production traces and one CONTROL. `synthetic` is Mooncake's third FAST'25 workload, built
# from public datasets with Poisson arrivals; it is reported as a control, never pooled with the
# production pair. Mooncake's older `arxiv-trace` is NOT used: it is the toolagent requests again
# (identical output_length at all 23,608 positions, block counts at 98.9%), re-timed and
# re-tokenized, so counting it as a third workload would count toolagent twice. See §AO.
TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl",
          "synthetic": "data/mooncake_synthetic_trace.jsonl"}


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

        # ONE instant for everything (§AP). The train/eval cut used to be the median block
        # arrival while costs were billed at time_split(0.5); they differed by 641 and 1,017
        # requests. Now the cut IS the billing split, so the evaluation population is exactly
        # the blocks with no pre-split access -- the population §5.2 says carries most of the
        # oracle's advantage.
        st = max(1, wl.time_split(0.5))
        tr_idx, ev_idx = split_by_arrival(arrival, st)
        if len(tr_idx) < 100 or len(ev_idx) < 100:
            print(f"{name}: too few arrivals to split ({len(tr_idx)}/{len(ev_idx)})")
            continue

        # Training labels that are fully observed before the split (§AP). They used to count
        # every later access, so 34.3% / 46.3% of training-label mass came from the evaluation
        # period. A fixed window keeps the label's meaning the same for every row.
        steps_of = {b: wl.traces[b].access_steps for b in ids}
        window = st // 2
        fit_idx, fit_lab = windowed_training_labels(arrival, ids, steps_of, st, window)
        assert all(arrival[i] + window < st for i in fit_idx), "a training label crosses the split"
        w = fit_logistic([X[i] for i in fit_idx], [1.0 if v > 0 else 0.0 for v in fit_lab])
        scores = predict(w, [X[i] for i in ev_idx])
        # AUC is scored against the evaluation blocks' true future -- that is the test truth, and
        # it never reaches the model or the beliefs.
        a_auc = auc(scores, [1.0 if y[i] > 0 else 0.0 for i in ev_idx])

        ev_ids = [ids[i] for i in ev_idx]
        fut = wl.future_counts(st)
        # Magnitudes a causal policy can use (§AP): the previous period's equivalent of the
        # evaluation population -- blocks first seen before the split, counted up to it, i.e.
        # their prefix counts. These used to be the evaluation blocks' TRUE future counts.
        pre = wl.prefix_stats(st)
        ref = analogue_magnitudes((v["count"] for v in pre.values()), len(ev_ids))

        c_pref = prefix_greedy_cost(wl)
        c_orac = place_with_beliefs(
            wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
        # Blocks that already exist at the split keep the frequency heuristic's belief in EVERY
        # arm, the ceiling included; only the new blocks' beliefs vary (§AP). The first causal
        # version gave pre-split blocks belief 0 in the arrival arms while the baseline placed them
        # by frequency, so every arrival arm discarded knowledge the baseline kept -- small on the
        # production traces, where those blocks carry ~5% of the gap, but enough on the synthetic
        # control to make even a clairvoyant arrival policy lose to the heuristic.
        # This mirrors the split-instant ceiling exactly: there, visible blocks are clairvoyant and
        # new blocks sit at their heuristic belief (0, having no prefix); here the roles swap.
        base = {i.item_id: float(pre[i.item_id]["count"]) for i in wl.items}

        def with_new(beliefs_for_new):
            b = dict(base)
            b.update(beliefs_for_new)
            return b

        c_ceil = place_with_beliefs(wl, with_new({i: float(fut.get(i, 0)) for i in ev_ids}))
        span, reach = c_pref - c_orac, c_pref - c_ceil

        def score(vals):
            belief = with_new(dict(zip(ev_ids, quantile_match(list(vals), ref))))
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
                                 "context; train/eval cut = billing split = time_split(0.5); "
                                 "training labels windowed to end before the split; belief "
                                 "magnitudes from the pre-split analogue population"),
                    "why": ("the split-instant protocol leaves ~95% of the hindsight gap "
                            "unreachable; this targets it, causally (§AP)")},
         "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
