#!/usr/bin/env python3
"""H2: does a block's age decide whether it is dead? (§BF; ADVISE H2)

The seen-once rule parks blocks seen once before the split behind the new ones. A pilot (ADVISE §2.2)
found age -- the time since a visible block's last access before the split -- a far sharper sign of
deadness than its count. This script adds causal arms that use age, all billed at the true cost (tier
only), with the thresholds taken from the prefix alone:

  primary (registered before running)
    old-low-count-last   count <= 2 and age above the median age of visible blocks -> behind new blocks
  exploratory (reported as such)
    old-last             age above the median, any count
    old75-last           age above the 25th percentile (the oldest three quarters), any count
    old-seen-once-last   count == 1 and age above the median

Pre-registered (ADVISE H2): confirmed if the primary arm beats the seen-once rule by 10 points or more on
both production traces; rejected if it gains less than 3, or does worse than the seen-once rule on two or
more of the four Alibaba traces.
Usage: python3 scripts/run_age_rules.py [--bailian]
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

from run_evidence_ladder import BAILIAN, TIER, TRACES, ordered  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402


def run(name, wl, split=0.5):
    st = max(1, wl.time_split(split))
    fut, pre = wl.future_counts(st), wl.prefix_stats(st)
    vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
    cnt = {b: float(pre[b]["count"]) for b in vis}
    age = {}
    for b in vis:
        s = wl.traces[b].access_steps
        age[b] = st - s[bisect.bisect_left(s, st) - 1]
    ages = sorted(age.values())
    med, q25 = ages[len(ages) // 2], ages[len(ages) // 4]
    dead = {b for b in vis if fut.get(b, 0) == 0}
    sets = {"seen-once-last": {b for b in vis if cnt[b] == 1},
            "old-low-count-last": {b for b in vis if cnt[b] <= 2 and age[b] > med},
            "old-last": {b for b in vis if age[b] > med},
            "old75-last": {b for b in vis if age[b] > q25},
            "old-seen-once-last": {b for b in vis if cnt[b] == 1 and age[b] > med},
            "dead-known": dead}
    arms = {"heuristic": (wl, cnt),
            "oracle": (wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})}
    for k, s in sets.items():
        arms[k] = (ordered(wl, s), {b: (0.0 if b in s else c) for b, c in cnt.items()})
    sq.LEGAL_ACTIONS = TIER
    cost = {a: sq.place_with_beliefs(w_, b_, split_frac=split) for a, (w_, b_) in arms.items()}
    span = cost["heuristic"] - cost["oracle"]
    prec = {k: (round(len(s & dead) / len(s), 4) if s else None) for k, s in sets.items() if k != "dead-known"}
    rec = {k: round(len(s & dead) / len(dead), 4) for k, s in sets.items() if k != "dead-known"}
    row = {"trace": name, "split": split, "median_age_steps": med,
           "captured_pct": {a: round(100 * (cost["heuristic"] - c) / span, 2) for a, c in cost.items()},
           "precision_for_dead": prec, "recall_for_dead": rec,
           "n_parked": {k: len(s) for k, s in sets.items()}}
    print(name, row["captured_pct"], flush=True)
    return row


def _one(args):
    name, path = args
    import stoa.mooncake as _mc
    _mc.BLOCK_TOKENS = 16
    return run(name, load_mooncake(str(ROOT / path)))


def main() -> int:
    bail = "--bailian" in sys.argv
    if bail:
        from multiprocessing import Pool
        with Pool(len(BAILIAN)) as pool:
            rows = pool.map(_one, list(BAILIAN.items()))
    else:
        rows = [run(n, load_mooncake(str(ROOT / p))) for n, p in TRACES.items()]
    dest = ROOT / "experiments" / ("age_rules_bailian.json" if bail else "age_rules.json")
    dest.write_text(json.dumps({"config": {"split": 0.5, "actions": "tier only", "primary": "old-low-count-last",
                                           "registered": "confirmed if primary - seen-once >= 10 on both production "
                                                         "traces; rejected if < 3 or worse on >= 2 of 4 Alibaba"},
                                "rows": rows}, indent=2) + "\n")
    print("wrote", dest.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
