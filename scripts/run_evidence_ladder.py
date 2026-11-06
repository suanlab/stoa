#!/usr/bin/env python3
"""How much of the oracle's advantage is knowing which seen blocks are dead? (§BA)

§AZ found that the split-time ceiling is set by where blocks with no future evidence go: 93% of the
blocks visible at the split are never read again, and parking them last moves the ceiling from 20% to
78% (tier-only). This ladder separates what the hindsight oracle knows into parts, each billed at the
true post-split cost and reported as a captured share of the oracle's advantage over the frequency
heuristic:

  heuristic        prefix counts for visible blocks, new blocks last (0)
  seen-once-last   CAUSAL: as heuristic, but visible blocks seen exactly once in the prefix are ordered
                   behind the new blocks (the prefix says a new block is reused more than a seen-once
                   one; §V-C)
  dead-known       clairvoyant on deadness only: prefix counts for live visible blocks, dead visible
                   blocks behind the new blocks
  visible-dead-first  clairvoyant counts for visible blocks, dead ones AHEAD of new blocks on the
                   belief-0 tie: the ceiling as published (`attainable_ceiling`)
  visible-dead-last   clairvoyant counts for visible blocks, dead ones behind new blocks (§AZ)
  oracle           clairvoyant counts for every block (= 100%)

Ordering is set by item order on the belief tie, so "behind the new blocks" means belief 0 and later
item order (the greedy pass is stable). Also reported: the share of visible blocks that are dead, and
how well "seen once" predicts "dead" (precision, recall). Splits 0.3-0.7, both action spaces.
Usage: python3 scripts/run_evidence_ladder.py [--bailian]
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_arrival_admission import TRACES  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa.environment import Representation, Timing  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

ALL = list(sq.LEGAL_ACTIONS)
TIER = [a for a in ALL if a.representation == Representation.PLAINTEXT and a.timing == Timing.ONLINE]
SPACES = {"tier": TIER, "all": ALL}
SPLITS = (0.3, 0.4, 0.5, 0.6, 0.7)
# --bailian: Alibaba's four Qwen traces (16-token blocks, converted by run_bailian.py), split 0.5 only
BAILIAN = {n: f"data/bailian_{n}_mooncake.jsonl" for n in ("traceA", "traceB", "coder", "thinking")}


def ordered(wl, last: set):
    """`wl` with the blocks in `last` moved to the end of item order (behind every other block on a
    belief tie), so the stable greedy pass reaches them after the new blocks."""
    return dataclasses.replace(wl, items=[i for i in wl.items if i.item_id not in last]
                               + [i for i in wl.items if i.item_id in last])


def _one(args):
    """One trace (Bailian mode, run in a worker process): the rows for its splits and spaces."""
    name, path, splits, spaces = args
    import stoa.mooncake as _mc
    _mc.BLOCK_TOKENS = 16
    return _rows(name, load_mooncake(str(ROOT / path)), splits, {s: SPACES[s] for s in spaces})


def _rows(name, wl, splits, spaces):
    rows = []
    try:
        for split in splits:
            st = max(1, wl.time_split(split))
            fut, pre = wl.future_counts(st), wl.prefix_stats(st)
            vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
            dead = {b for b in vis if fut.get(b, 0) == 0}
            once = {b for b in vis if int(pre[b]["count"]) == 1}
            cnt = {b: float(pre[b]["count"]) for b in vis}
            arms = {
                "heuristic": (wl, cnt),
                "seen-once-last": (ordered(wl, once), {b: (0.0 if b in once else c) for b, c in cnt.items()}),
                "dead-known": (ordered(wl, dead), {b: (0.0 if b in dead else c) for b, c in cnt.items()}),
                "visible-dead-first": (wl, {b: float(fut.get(b, 0)) for b in vis}),
                "visible-dead-last": (ordered(wl, dead), {b: float(fut.get(b, 0)) for b in vis}),
                "oracle": (wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items}),
            }
            for space, acts in spaces.items():
                sq.LEGAL_ACTIONS = acts
                cost = {a: sq.place_with_beliefs(w_, bel, split_frac=split) for a, (w_, bel) in arms.items()}
                span = cost["heuristic"] - cost["oracle"]
                rows.append({"trace": name, "split": split, "actions": space,
                             "n_visible": len(vis), "dead_visible_pct": round(100 * len(dead) / len(vis), 2),
                             "seen_once_precision_for_dead": round(len(once & dead) / len(once), 4) if once else None,
                             "seen_once_recall_for_dead": round(len(once & dead) / len(dead), 4) if dead else None,
                             "captured_pct": {a: round(100 * (cost["heuristic"] - c) / span, 2)
                                              for a, c in cost.items()}})
                print(name, split, space, rows[-1]["dead_visible_pct"], rows[-1]["captured_pct"], flush=True)
    finally:
        sq.LEGAL_ACTIONS = ALL
    return rows


def main() -> int:
    bail = "--bailian" in sys.argv
    if bail:
        # tier-only (the paper's primary space): the 50-action space is impractically slow on traces
        # of 16-token blocks. One worker process per trace.
        from multiprocessing import Pool
        with Pool(len(BAILIAN)) as pool:
            parts = pool.map(_one, [(n, p, (0.5,), ("tier",)) for n, p in BAILIAN.items()])
        rows = [r for part in parts for r in part]
        (ROOT / "experiments" / "evidence_ladder_bailian.json").write_text(json.dumps(
            {"config": {"splits": [0.5], "traces": sorted(BAILIAN), "actions": "tier only",
                        "block_tokens": 16,
                        "share": "(heuristic - arm) / (heuristic - oracle), true bill"},
             "rows": rows}, indent=2) + "\n")
        print("wrote evidence_ladder_bailian.json")
        return 0
    rows = []
    for name, path in TRACES.items():
        rows += _rows(name, load_mooncake(str(ROOT / path)), SPLITS, SPACES)
    (ROOT / "experiments" / "evidence_ladder.json").write_text(json.dumps(
        {"config": {"splits": list(SPLITS), "share": "(heuristic - arm) / (heuristic - oracle), true bill",
                    "behind": "belief 0 and later item order than every new block (stable greedy)"},
         "rows": rows}, indent=2) + "\n")
    print("wrote experiments/evidence_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
