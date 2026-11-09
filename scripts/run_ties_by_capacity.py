#!/usr/bin/env python3
"""Does the tie rule's effect depend on the tier capacities? (§BI)

Every placement result uses one capacity setting: the GPU, host and remote tiers hold 10%, 30% and 60%
of the workload's bytes, disk the rest. An independent review asked whether the tie rule's 48--63 point
swing is an artifact of that choice. This script scales the three finite tiers together by a factor
(disk stays unbounded) and reruns, tier-only at split 0.5, on the production traces and the synthetic
control:

  heuristic        prefix counts, new blocks last (0)
  seen-once-last   causal: visible blocks seen once behind the new blocks
  dead-first       clairvoyant visible futures, dead visible blocks ahead of new ones (the published rule)
  dead-last        the same, dead visible blocks behind new ones
  oracle           clairvoyant futures for every block (100)

Also reported: the oracle's advantage over the heuristic in ms of block reads per post-split request
(cost / lambda_latency; the token term is the same on every tier and cancels, tier-only actions change
no representation), so the shares can be read against the size of what they share.
Usage: python3 scripts/run_ties_by_capacity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_evidence_ladder import TIER, ordered  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa import simulator as simmod  # noqa: E402
from stoa.environment import Tier  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from train_session_features import parse_requests  # noqa: E402

TRACES = {n: f"data/mooncake_{n}_trace.jsonl" for n in ("conversation", "toolagent", "synthetic")}
SCALES = (0.25, 0.5, 1.0, 1.5)
FINITE = (Tier.GPU, Tier.CPU, Tier.REMOTE_RDMA)


def ladder(wl, path):
    st = max(1, wl.time_split(0.5))
    fut, pre = wl.future_counts(st), wl.prefix_stats(st)
    vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
    dead = {b for b in vis if fut.get(b, 0) == 0}
    once = {b for b in vis if int(pre[b]["count"]) == 1}
    cnt = {b: float(pre[b]["count"]) for b in vis}
    arms = {"heuristic": (wl, cnt),
            "seen-once-last": (ordered(wl, once), {b: (0.0 if b in once else c) for b, c in cnt.items()}),
            "dead-first": (wl, {b: float(fut.get(b, 0)) for b in vis}),
            "dead-last": (ordered(wl, dead), {b: float(fut.get(b, 0)) for b in vis}),
            "oracle": (wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})}
    cost = {a: sq.place_with_beliefs(w_, b_) for a, (w_, b_) in arms.items()}
    span = cost["heuristic"] - cost["oracle"]
    n_req = sum(1 for t, _s, _b in parse_requests(str(ROOT / path), 10 ** 9) if t >= st)
    return ({a: round(100 * (cost["heuristic"] - c) / span, 2) for a, c in cost.items()},
            round(span / DEFAULT_WEIGHTS.lambda_latency / n_req, 3), n_req)


def main() -> int:
    base = dict(simmod._TIER_CAPACITY_FRACTION)
    rows = []
    sq.LEGAL_ACTIONS = TIER
    try:
        for name, path in TRACES.items():
            wl = load_mooncake(str(ROOT / path))
            for s in SCALES:
                for t in FINITE:
                    simmod._TIER_CAPACITY_FRACTION[t] = base[t] * s
                pct, ms, n_req = ladder(wl, path)
                rows.append({"trace": name, "scale": s,
                             "fractions": {t.value: round(simmod._TIER_CAPACITY_FRACTION[t], 4) for t in FINITE},
                             "captured_pct": pct, "tie_rule_swing": round(pct["dead-last"] - pct["dead-first"], 2),
                             "oracle_advantage_ms_per_request": ms, "post_split_requests": n_req})
                print(rows[-1], flush=True)
    finally:
        simmod._TIER_CAPACITY_FRACTION.clear()
        simmod._TIER_CAPACITY_FRACTION.update(base)
    (ROOT / "experiments" / "ties_by_capacity.json").write_text(json.dumps(
        {"config": {"split": 0.5, "actions": "tier only", "scales": list(SCALES),
                    "base_fractions": {t.value: base[t] for t in FINITE},
                    "ms": "(heuristic - oracle) cost / lambda_latency / post-split requests, declared read table"},
         "rows": rows}, indent=2) + "\n")
    print("wrote experiments/ties_by_capacity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
