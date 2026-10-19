#!/usr/bin/env python3
"""Place blocks when they ARRIVE, not at a split instant.

A split-instant policy can only place blocks that exist at the split, and on these traces those
carry ~5% of the hindsight oracle's advantage (§5.2). Here every block appearing for the FIRST
time after the split is placed at that instant, from request-level context available then
(session request count, rate and recency; position in the request; request length; distinct
blocks seen so far). No feature reads the future.

Blocks that exist at the split keep the frequency heuristic's belief in every arm; only the new
blocks' beliefs differ. A belief has two parts, and the arms vary them separately (§AQ):

  ORDER, on one common magnitude multiset (the pre-split analogue, quantile-matched):
    order-arrival    earliest arrival highest (first-come-first-served)
    order-learned    a logistic model trained on windowed pre-split labels
    order-shuffled   a seeded random order
  MAGNITUDE, in arrival order (a constant belief; the stable sort keeps arrival order):
    constant-mean    every new block at the mean pre-split reuse
    constant-median  every new block at the median pre-split reuse

An earlier version compared a constant-mean FCFS arm against learned and shuffled arms on the
spread multiset, so its "order" comparison was mostly a magnitude comparison (§AQ).

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
from stoa.credit import weighted_cost  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.gnn import LEGAL_ACTIONS  # noqa: E402  (after run_tier_only's patch, when it applies)
from stoa.simulator import TieringSimulator  # noqa: E402

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


TIES = ("new-behind", "new-ahead")
TIE_EPS = 1e-9


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

        # TIE CONVENTION (§AR). place_with_beliefs sorts stably and every pre-split item precedes
        # every new one, so at EQUAL belief a new block is placed BEHIND the pre-split block --
        # a convention, not a measurement, and it moved the order arms by ~20 points and the
        # mean/median gap by up to 39. Both conventions are reported. "new-ahead" adds TIE_EPS to
        # every new-block belief: enough to win a tie, and asserted below never to change which
        # action a belief selects.
        sim = TieringSimulator()
        rep = wl.items[0].representation

        def _argmin_action(v):
            return min(LEGAL_ACTIONS, key=lambda a_: (weighted_cost(
                sim.cost_of_placement(a_, v, rep), DEFAULT_WEIGHTS), sim.read_latency(a_.tier)))

        def bill(belief_for_new, tie):
            if tie == "new-ahead":
                vals = set(belief_for_new.values())
                assert all(_argmin_action(v) == _argmin_action(v + TIE_EPS) for v in vals), \
                    "TIE_EPS changed an action choice"
                belief_for_new = {k: v + TIE_EPS for k, v in belief_for_new.items()}
            c = place_with_beliefs(wl, with_new(belief_for_new))
            return (100.0 * (c_pref - c) / span if span > 0 else 0.0,
                    100.0 * (c_pref - c) / reach if reach > 0 else 0.0)

        # ORDER arms: every arm's scores are quantile-matched onto the SAME multiset, so the only
        # thing that differs between them is which new block receives which magnitude. Strict
        # scores throughout -- a tied score would receive the tie block's mean and smuggle a
        # magnitude difference back in, which is exactly the defect §AQ records.
        import random as _r
        import statistics as _st
        perm = list(range(len(ev_idx)))
        _r.Random(17).shuffle(perm)
        assert all(arrival[ev_idx[k]] <= arrival[ev_idx[k + 1]] for k in range(len(ev_idx) - 1)), \
            "evaluation blocks are not in arrival order"
        # Learned scores can tie (identical feature vectors); rank them with arrival as the
        # tie-break so every arm hands out the multiset one value per block.
        _ord = sorted(range(len(ev_idx)), key=lambda k: (scores[k], -k))
        learned_rank = [0.0] * len(ev_idx)
        for rnk, k in enumerate(_ord):
            learned_rank[k] = float(rnk)
        order_arms = {
            "order-arrival": [-float(k) for k in range(len(ev_idx))],   # ev_idx is in arrival order
            "order-learned": learned_rank,
            "order-shuffled": [float(v) for v in perm],
        }
        # MAGNITUDE arms: one constant belief for every new block. place_with_beliefs sorts
        # stably, so equal beliefs are taken in item order, i.e. arrival order.
        mag_arms = {"constant-mean": _st.mean(ref), "constant-median": _st.median(ref)}

        for tie in TIES:
            for arm, vals in order_arms.items():
                au = a_auc if arm == "order-learned" else None   # reported only; truth-derived
                g, r = bill(dict(zip(ev_ids, quantile_match(list(vals), ref))), tie)
                print(f"{name:<13} | {arm:<17} | {au if au is not None else float('nan'):6.3f} | "
                      f"{g:7.1f}% | {r:8.1f}%   [{tie}]")
                rows.append({"trace": name, "axis": "order", "tie": tie, "arm": arm,
                             "auc": round(au, 4) if au is not None else None,
                             "captured_pct": round(g, 2),
                             "captured_of_attainable_pct": round(r, 2),
                             "n_arrivals_scored": len(ev_idx)})
            for arm, m in mag_arms.items():
                g, r = bill({i: m for i in ev_ids}, tie)
                print(f"{name:<13} | {arm:<17} | {'':>6} | {g:7.1f}% | {r:8.1f}%   "
                      f"(belief {m:.2f}) [{tie}]")
                rows.append({"trace": name, "axis": "magnitude", "tie": tie, "arm": arm,
                             "auc": None, "new_block_belief": round(m, 4),
                             "captured_pct": round(g, 2),
                             "captured_of_attainable_pct": round(r, 2),
                             "n_arrivals_scored": len(ev_idx)})
        ceil_g = 100.0 * reach / span if span > 0 else 0.0
        print(f"{name:<13} | {'arrival-ceiling':<17} | {1.0:6.3f} | {ceil_g:7.1f}% | {100.0:8.1f}%")
        rows.append({"trace": name, "axis": "diagnostic", "arm": "arrival-ceiling", "auc": 1.0,
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
                                 "magnitudes from the pre-split analogue population; order arms share "
                                 "one magnitude multiset, magnitude arms share arrival order"),
                    "why": ("the split-instant protocol leaves ~95% of the hindsight gap "
                            "unreachable; this targets it, causally (§AP)")},
         "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
