#!/usr/bin/env python3
"""H8: is a block's session a better unit than the block? (§BH; ADVISE H8)

§BF found that age marks dead blocks precisely but reaches few of them. KV blocks are created and abandoned
with sessions, and the traces carry no session ids. The root of a block's hash chain is not one: on
conversation every request starts with the same system-prompt block. A session is approximated from the
prefix alone instead: a block is a branching point if, before the split, it was followed by two or more
different blocks (the end of a shared system prompt, or a fork); a block's session is the block right after
the first branching point on its chain, the start of its conversation-specific part. A block before any
branching point keeps its own identity. Session age at the split is the time since the session block was
last read, which bounds every later block of that session, since reading a block reads its prefix. Arms (tier-only, split 0.5, billed at the true cost), each parking
visible blocks behind new ones:

  block-old-last     the block's own age above the median block age      (run_age_rules.py's old-last)
  session-old-last   the session's age above the median session age, over visible blocks

Pre-registered (ADVISE H8): not rejected if session-old-last is no more than 3 points below block-old-last
on both production traces; rejected if it is 3 or more points below on either.
Usage: python3 scripts/run_session_age.py
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

from run_evidence_ladder import TIER, ordered  # noqa: E402
from stoa import sequential as sq  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

TRACES = {n: f"data/mooncake_{n}_trace.jsonl" for n in ("conversation", "toolagent", "synthetic")}


def sessions(path, st):
    """Session block of every block, from the requests before the split (causal)."""
    reqs = []
    with open(ROOT / path) as fh:
        for line in fh:
            reqs.append(json.loads(line).get("hash_ids", []))
    succ = {}
    for hs in reqs[:st]:
        for a, b in zip(hs, hs[1:]):
            succ.setdefault(a, set()).add(b)
    branching = {h for h, s in succ.items() if len(s) >= 2}
    session = {}
    for hs in reqs[:st]:
        first = next((i for i, h in enumerate(hs[:-1]) if h in branching), None)
        for j, h in enumerate(hs):
            b = f"b{h}"
            if b in session:
                continue
            session[b] = f"b{hs[first + 1]}" if first is not None and j > first else b
    return session, len(branching)


def main() -> int:
    rows = []
    sq.LEGAL_ACTIONS = TIER
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        st = max(1, wl.time_split(0.5))
        root, n_branch = sessions(path, st)
        fut, pre = wl.future_counts(st), wl.prefix_stats(st)
        vis = [i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st]
        cnt = {b: float(pre[b]["count"]) for b in vis}

        def age(b):
            s = wl.traces[b].access_steps
            return st - s[bisect.bisect_left(s, st) - 1]
        a_blk = {b: age(b) for b in vis}
        a_ses = {b: age(root[b]) for b in vis}
        med_b = sorted(a_blk.values())[len(vis) // 2]
        med_s = sorted(a_ses.values())[len(vis) // 2]
        dead = {b for b in vis if fut.get(b, 0) == 0}
        sets = {"seen-once-last": {b for b in vis if cnt[b] == 1},
                "block-old-last": {b for b in vis if a_blk[b] > med_b},
                "session-old-last": {b for b in vis if a_ses[b] > med_s}}
        arms = {"heuristic": (wl, cnt), "oracle": (wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})}
        for k, s in sets.items():
            arms[k] = (ordered(wl, s), {b: (0.0 if b in s else c) for b, c in cnt.items()})
        cost = {a: sq.place_with_beliefs(w_, b_) for a, (w_, b_) in arms.items()}
        span = cost["heuristic"] - cost["oracle"]
        row = {"trace": name, "branching_blocks": n_branch,
               "n_sessions_visible": len({root[b] for b in vis}),
               "captured_pct": {a: round(100 * (cost["heuristic"] - c) / span, 2) for a, c in cost.items()},
               "precision_for_dead": {k: round(len(s & dead) / len(s), 4) for k, s in sets.items()},
               "recall_for_dead": {k: round(len(s & dead) / len(dead), 4) for k, s in sets.items()}}
        rows.append(row)
        print(name, row["captured_pct"], row["precision_for_dead"], row["recall_for_dead"], "branching", n_branch, "sessions", row["n_sessions_visible"],
              flush=True)
    (ROOT / "experiments" / "session_age.json").write_text(json.dumps(
        {"config": {"split": 0.5, "actions": "tier only",
                    "session": "block after the first branching point on the chain (branching = >= 2 successors "
                               "seen before the split)",
                    "registered": "rejected if session-old-last is >= 3 points below block-old-last on either "
                                  "production trace"}, "rows": rows}, indent=2) + "\n")
    print("wrote experiments/session_age.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
