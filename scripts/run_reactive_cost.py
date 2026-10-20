#!/usr/bin/env python3
"""Reactive admission billed on the arrival-time metric: is "rank unseen blocks high" just recency?

§V-C measures arrival-time placement as a share of the hindsight oracle's cost advantage; §V-F
measures reactive caching as a share of Belady's hit rate. The final pre-submission review asked for
one metric. Here multi-tier reactive policies replay the post-split trace from the same starting
state as every arrival arm (pre-split blocks placed by the frequency heuristic), with tier-only
actions, the same per-access bill and the same migration prices as run_online_admission.py:

  none        no moves: each new block takes the fastest tier with room at its creation. Must equal
              the static frequency heuristic exactly (asserted) -- the validation of this replay.
  lru         multi-tier LRU: every access, creations included, moves the block to the fastest
              tier; overflow cascades down one tier at a time.
  second-hit  as lru, but a block's creation places it in the slowest tier (cache on second hit,
              Maggs & Sitaraman); its first reuse promotes it.
Captured share = (heuristic cost - policy cost) / (heuristic cost - oracle cost), creations unbilled.
Usage: python3 scripts/run_reactive_cost.py
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

import run_online_admission as oa  # noqa: E402
from stoa.environment import Action  # noqa: E402

TIER_ONLY = oa.tier_only_actions()
Action.space = staticmethod(lambda: list(TIER_ONLY))

from run_arrival_admission import TRACES  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

MODELS = ("zero", "read+write")


def replay(policy, init_tier, is_new, slots, cm, events, w):
    tiers = cm["tiers"]
    k = len(tiers)
    res = [OrderedDict() for _ in range(k)]              # per tier, LRU first
    where = {}
    for b, t in init_tier.items():
        res[t][b] = None
        where[b] = t
    moves = {m: 0.0 for m in MODELS}

    def move(b, src, dst):
        for m in MODELS:
            moves[m] += oa.migration_cost(tiers[src], tiers[dst], cm["read_ms"], w, m)

    def insert_top(b, src):
        """b (not resident in tier 0) goes to tier 0; overflow cascades down to `src` (or the end)."""
        res[0][b] = None
        where[b] = 0
        if src is not None:
            move(b, src, 0)
        i = 0
        while len(res[i]) > slots[i]:
            v, _ = res[i].popitem(last=False)
            res[i + 1][v] = None
            where[v] = i + 1
            move(v, i, i + 1)
            i += 1

    def place_free(b, fastest):
        order = range(k) if fastest else range(k - 1, -1, -1)
        for i in order:
            if len(res[i]) < slots[i]:
                res[i][b] = None
                where[b] = i
                return
        raise AssertionError("no tier has room")

    cost = 0.0
    for e in events:
        b = e if e >= 0 else ~e
        if e < 0:                                          # creation: compulsory miss, not billed
            if policy == "none":
                place_free(b, fastest=True)
            elif policy == "lru":
                insert_top(b, None)
            elif policy == "second-hit":
                place_free(b, fastest=False)
            continue
        t = where[b]
        cost += cm["per"][tiers[t]]
        if policy in ("lru", "second-hit") and t != 0:
            del res[t][b]
            insert_top(b, t)
        elif policy in ("lru", "second-hit"):
            res[0].move_to_end(b)
    n_blocks = len(where)
    fixed = cm["fixed"][tiers[0]] * n_blocks
    return {m: cost + fixed + moves[m] for m in MODELS}


def main() -> int:
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    online = json.loads((ROOT / "experiments" / "online_admission.json").read_text())
    rows = []
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        st = max(1, wl.time_split(0.5))
        pre, fut = wl.prefix_stats(st), wl.future_counts(st)
        ids = [i.item_id for i in wl.items]
        index = {b: j for j, b in enumerate(ids)}
        sizes = [float(i.size_bytes) for i in wl.items]
        cm = oa.cost_model(sim, w, TIER_ONLY, wl.items[0].representation)
        caps = sim.tier_capacities(wl.total_bytes)
        # A tier may be unbounded (inf bytes); no tier needs more slots than there are blocks.
        slots = [len(ids) if caps[t] == float("inf") else int(caps[t] // sizes[0]) for t in cm["tiers"]]
        events = oa.post_split_events(str(ROOT / path), st, 10 ** 9, index)
        counts = [float(pre[b]["count"]) for b in ids]
        is_new = [c == 0 for c in counts]
        heur_tiers = oa.static_tiers(oa.rank_order(counts, is_new, False), sizes, caps, cm["tiers"])
        c_heur = oa.bill_static(events, heur_tiers, cm)
        assert abs(c_heur - place_with_beliefs(wl, {b: counts[j] for j, b in enumerate(ids)})) <= 1e-6 * c_heur
        c_orac = place_with_beliefs(wl, {b: float(fut.get(b, 0)) for b in ids})
        span = c_heur - c_orac
        tidx = {t: j for j, t in enumerate(cm["tiers"])}
        init = {j: tidx[heur_tiers[j]] for j in range(len(ids)) if not is_new[j]}
        out = {}
        for pol in ("none", "lru", "second-hit"):
            c = replay(pol, init, is_new, slots, cm, events, w)
            out[pol] = {m: round(100.0 * (c_heur - c[m]) / span, 2) for m in MODELS}
        assert abs(out["none"]["zero"]) < 0.01, f"replay does not reproduce the heuristic: {out['none']}"
        best_arm = {m: max(r["online_captured_pct"] for r in online["rows"]
                           if r["trace"] == name and r["migration_cost_model"] == m
                           and r["policy"] == "onestep" and r["arm"] != "arrival-ceiling") for m in MODELS}
        rows.append({"trace": name, "reactive": out, "best_online_arrival_arm": best_arm})
        print(name, out, "best online arrival arm", best_arm, flush=True)
    (ROOT / "experiments" / "reactive_cost.json").write_text(json.dumps(
        {"config": {"actions": "tier only", "start": "pre-split blocks by the frequency heuristic",
                    "migration": "zero / read+write per move (as run_online_admission.py)",
                    "bill": "creations are compulsory misses"}, "rows": rows}, indent=2) + "\n")
    print("wrote experiments/reactive_cost.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
