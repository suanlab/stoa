#!/usr/bin/env python3
"""H3 and H4: move a block only when it has paid for the move; predict where moving stops paying (§BG).

Every policy in the online replays that moves blocks loses to the static frequency heuristic once a move is
priced, and LRU, which promotes a block on its first access from a slow tier, loses most. Promotion is a
rent-or-buy decision: each access from tier t costs per[t] - per[gpu] more than it would from the GPU, and
a move costs a fixed price. Ski rental (Karlin et al. 1988) buys once the rent paid reaches the price,
which is 2-competitive per decision. This script replays, from the heuristic's starting state with
tier-only actions on the production traces:

  ski          creations take the fastest tier with room (as the heuristic); a resident on a slower tier
               accumulates the extra cost of its accesses and is promoted to the GPU tier once that reaches
               alpha x the read+write price of the move; the GPU tier is LRU-ordered and overflow cascades
               down one tier at a time, each demotion billed at alpha x its read+write price
  lru          run_reactive_cost.py's multi-tier LRU, billed at the same alpha (linear in alpha)

over a log grid of alpha (the move price as a multiple of one read plus one write; the policy knows it).

H3, registered before running: confirmed if ski captures >= 0% at every grid price inside the measured band
(0.75-1.87 from the block-move timings of notebook AX; 0.77 in the serving stack of notebook BE); rejected if it is below -5% at every
grid price in the band.

H4, registered before running: the price at which a policy stops beating the heuristic can be predicted
from the past. Replay the same policy on the prefix alone (the trace cut at the split, pseudo-split at its
midpoint) and take the alpha where its captured share crosses zero; confirmed if that prediction is within
+-30% of the alpha observed after the real split, for LRU and for ski, on both production traces.
Exploratory, added after H3 was rejected: `ski_cascade`, whose threshold also counts the cascade of
demotions each promotion forces (62% of ski's move cost at alpha = 1).
Usage: python3 scripts/run_ski_rental.py
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_reactive_cost as rc  # noqa: E402  (tier-only Action.space)
import run_online_admission as oa  # noqa: E402
from run_horizon import truncated  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.87, 2.0, 3.0)
BAND = (0.75, 1.87)


def ski_replay(alpha, init_tier, slots, cm, events, w, cascade=False):
    """Returns (access cost + fixed, move cost at alpha). With `cascade` (exploratory, chosen after H3
    was rejected) the threshold also counts the demotions the promotion forces: a block leaving tier t
    frees a slot there, so the cascade is 0->1, ..., (t-1)->t."""
    tiers = cm["tiers"]
    k = len(tiers)
    per = [cm["per"][t] for t in tiers]
    mig = lambda s, d: alpha * oa.migration_cost(tiers[s], tiers[d], cm["read_ms"], w, "read+write")  # noqa: E731
    res = [OrderedDict() for _ in range(k)]
    where, extra = {}, {}
    for b, t in init_tier.items():
        res[t][b] = None
        where[b] = t
    moves = 0.0
    tally = {"promotions": 0, "demotions": 0, "promotion_cost": 0.0, "demotion_cost": 0.0}

    def to_top(b, src):
        nonlocal moves
        del res[src][b]
        res[0][b] = None
        where[b] = 0
        moves += mig(src, 0)
        tally["promotions"] += 1
        tally["promotion_cost"] += mig(src, 0)
        i = 0
        while len(res[i]) > slots[i]:
            v, _ = res[i].popitem(last=False)
            res[i + 1][v] = None
            where[v] = i + 1
            extra[v] = 0.0
            moves += mig(i, i + 1)
            tally["demotions"] += 1
            tally["demotion_cost"] += mig(i, i + 1)
            i += 1

    cost = 0.0
    for e in events:
        b = e if e >= 0 else ~e
        if e < 0:
            for i in range(k):
                if len(res[i]) < slots[i]:
                    res[i][b] = None
                    where[b] = i
                    break
            continue
        t = where[b]
        cost += per[t]
        if t == 0:
            res[0].move_to_end(b)
            continue
        extra[b] = extra.get(b, 0.0) + per[t] - per[0]
        price = mig(t, 0) + (sum(mig(i, i + 1) for i in range(t)) if cascade else 0.0)
        if extra[b] >= price:
            extra[b] = 0.0
            to_top(b, t)
    fixed = cm["fixed"][tiers[0]] * len(where)
    ski_replay.last_tally = tally
    return cost + fixed, moves


def setup(wl, path, split_frac, max_requests=10 ** 9):
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    st = max(1, wl.time_split(split_frac))
    pre, fut = wl.prefix_stats(st), wl.future_counts(st)
    ids = [i.item_id for i in wl.items]
    index = {b: j for j, b in enumerate(ids)}
    sizes = [float(i.size_bytes) for i in wl.items]
    cm = oa.cost_model(sim, w, rc.TIER_ONLY, wl.items[0].representation)
    caps = sim.tier_capacities(wl.total_bytes)
    slots = [len(ids) if caps[t] == float("inf") else int(caps[t] // sizes[0]) for t in cm["tiers"]]
    events = oa.post_split_events(str(ROOT / path), st, max_requests, index)
    counts = [float(pre[b]["count"]) for b in ids]
    is_new = [c == 0 for c in counts]
    heur = oa.static_tiers(oa.rank_order(counts, is_new, False), sizes, caps, cm["tiers"])
    c_heur = oa.bill_static(events, heur, cm)
    span = c_heur - place_with_beliefs(wl, {b: float(fut.get(b, 0)) for b in ids}, split_frac=split_frac)
    tidx = {t: j for j, t in enumerate(cm["tiers"])}
    init = {j: tidx[heur[j]] for j in range(len(ids)) if not is_new[j]}
    return dict(w=w, cm=cm, slots=slots, events=events, is_new=is_new, init=init, c_heur=c_heur, span=span)


def curves(S):
    """Captured share of ski (at each alpha, priced at that alpha) and LRU (linear in alpha)."""
    lru = rc.replay("lru", S["init"], S["is_new"], S["slots"], S["cm"], S["events"], S["w"])
    lru_free, lru_rw = lru["zero"], lru["read+write"]
    out = []
    for a in ALPHAS:
        acc, mv = ski_replay(a, S["init"], S["slots"], S["cm"], S["events"], S["w"])
        ski = 100 * (S["c_heur"] - acc - mv) / S["span"]
        acc2, mv2 = ski_replay(a, S["init"], S["slots"], S["cm"], S["events"], S["w"], cascade=True)
        ski2 = 100 * (S["c_heur"] - acc2 - mv2) / S["span"]
        lr = 100 * (S["c_heur"] - (lru_free + a * (lru_rw - lru_free))) / S["span"]
        out.append({"alpha": a, "ski_pct": round(ski, 3), "ski_cascade_pct": round(ski2, 3), "lru_pct": round(lr, 3)})
    return out


def crossing(rows, key):
    """Smallest alpha at which the share goes from > 0 to <= 0 (linear interpolation), or None."""
    for r0, r1 in zip(rows, rows[1:]):
        if r0[key] > 0 >= r1[key]:
            return round(r0["alpha"] + (r1["alpha"] - r0["alpha"]) * r0[key] / (r0[key] - r1[key]), 4)
    return None


def main() -> int:
    out = {"config": {"alphas": list(ALPHAS), "band": list(BAND), "serving_alpha": 0.77,
                      "price": "alpha x read+write (run_online_admission.migration_cost)",
                      "registered_H3": "confirmed if ski >= 0 at every grid alpha in the band; rejected if < -5 at all",
                      "registered_H4": "prefix-replay break-even within +-30% of the observed one, LRU and ski, "
                                       "both traces"}, "traces": {}}
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        S = curves(setup(wl, path, 0.5))
        st = max(1, wl.time_split(0.5))
        pw = truncated(wl, st - 1)                      # the past only: requests before the split
        lo, hi = pw.accesses[0][0], pw.accesses[-1][0]
        P = curves(setup(pw, path, 0.5, max_requests=st))
        obs = {"lru": crossing(S, "lru_pct"), "ski": crossing(S, "ski_pct")}
        pred = {"lru": crossing(P, "lru_pct"), "ski": crossing(P, "ski_pct")}
        band = [r for r in S if BAND[0] <= r["alpha"] <= BAND[1]]
        S1 = setup(wl, path, 0.5)
        ski_replay(1.0, S1["init"], S1["slots"], S1["cm"], S1["events"], S1["w"])
        tl = dict(ski_replay.last_tally)
        tl["demotions_per_promotion"] = round(tl["demotions"] / tl["promotions"], 3)
        tl["demotion_share_of_move_cost"] = round(tl["demotion_cost"] / (tl["demotion_cost"] + tl["promotion_cost"]), 4)
        out["traces"][name] = {"after_split": S, "prefix_replay": P, "break_even_observed": obs,
                               "ski_cascade_min_in_band": min(r["ski_cascade_pct"] for r in band),
                               "ski_cascade_max_in_band": max(r["ski_cascade_pct"] for r in band),
                               "break_even_predicted": pred, "ski_moves_at_alpha_1": tl,
                               "ski_min_in_band": min(r["ski_pct"] for r in band),
                               "ski_max_in_band": max(r["ski_pct"] for r in band)}
        print(name, "observed", obs, "predicted", pred, "ski in band",
              out["traces"][name]["ski_min_in_band"], out["traces"][name]["ski_max_in_band"], flush=True)
        for r in S:
            print("   ", r, flush=True)
    (ROOT / "experiments" / "ski_rental.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/ski_rental.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
