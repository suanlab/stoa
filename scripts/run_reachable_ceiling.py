#!/usr/bin/env python3
"""The §5.2 table, computed in ONE frame, and swept over the split instant.

Why this script exists. Every number in §5.2 -- the section the paper's contribution rests on --
was absent from `experiments/` and asserted by nothing. Twelve re-verification passes built guards
around numbers that were in artifacts; the single number the paper turns on had never been in one.

Recomputing it showed the table mixed frames. Its `attainable gap` (4,996 / 3,900) matches
`prefix_greedy - attainable_ceiling` exactly, while its `full hindsight gap` (160,200 / 162,602) is
1.72x `prefix_greedy - oracle` on *both* traces. A ratio whose halves use different references is
the §W defect, and here it set the headline.

So: one reference, one run, every quantity derived from it. The reference is `prefix_greedy` -- the
frequency heuristic a causal policy is actually competing against -- because that is the frame the
attainable gap was already in.

The split sweep answers the obvious objection to the finding, in the same artifact: if almost all
of the gap is unreachable only at the split we happened to choose, the result is about our setup
rather than about the protocol. Run at 0.3-0.7 to show which it is.

    python3 scripts/run_reachable_ceiling.py              # full traces, splits 0.3-0.7
    python3 scripts/run_reachable_ceiling.py --requests 1500 --splits 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import (DEFAULT_WEIGHTS, attainable_ceiling,  # noqa: E402
                             place_with_beliefs)
from stoa.simulator import TieringSimulator  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}


def one(wl, split_frac: float, sim, w) -> dict:
    """All three costs at ONE split, through ONE routine.

    The first version of this function called `reference_costs` for the heuristic and the
    oracle and `attainable_ceiling` for the ceiling. `reference_costs` takes no `split_frac`
    -- it is always 0.5 -- so the numerator moved with the sweep and the denominator did not,
    and the attainable gap came out *negative* at split 0.3. The script written to fix a
    mixed-frame ratio mixed frames. It also passed a bare `RewardWeights()` where the module
    uses `DEFAULT_WEIGHTS`, whose `lambda_tokens` is 0.0005 rather than 1.0, inflating every
    cost by three orders of magnitude.

    Both are avoided the same way: every quantity below comes from `place_with_beliefs` at the
    same `split_frac`, differing only in the belief handed to it.
    """
    st = max(1, wl.time_split(split_frac))
    pre = wl.prefix_stats(st)
    fut = wl.future_counts(st)
    seen = [i.item_id for i in wl.items if pre.get(i.item_id, {}).get("count", 0) > 0]
    n_blocks = len(wl.items)

    # heuristic: the future looks like the observed prefix.
    heuristic = place_with_beliefs(
        wl, {i.item_id: float(pre.get(i.item_id, {}).get("count", 0)) for i in wl.items},
        sim, w, split_frac)
    # oracle: clairvoyant over every block, including ones not yet seen.
    oracle = place_with_beliefs(
        wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items}, sim, w, split_frac)
    # ceiling: clairvoyant, but only about blocks a causal policy could have an opinion on.
    ceiling = attainable_ceiling(wl, seen, sim, w, split_frac=split_frac)
    full = heuristic - oracle
    attainable = heuristic - ceiling
    return {
        "split_frac": split_frac,
        "n_blocks": n_blocks,
        "blocks_with_no_pre_split_access": n_blocks - len(seen),
        "no_pre_split_pct": round(100.0 * (n_blocks - len(seen)) / n_blocks, 1),
        # Every quantity below is a difference from the SAME reference.
        "reference": "prefix_greedy",
        "reference_cost": round(heuristic, 1),
        "oracle_cost": round(oracle, 1),
        "attainable_ceiling_cost": round(ceiling, 1),
        "full_hindsight_gap": round(full, 1),
        "attainable_gap": round(attainable, 1),
        "unreachable_pct": round(100.0 * (1.0 - attainable / full), 1) if full else None,
        "reachable_pct": round(100.0 * attainable / full, 1) if full else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--splits", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--out", type=str, default="experiments/reachable_ceiling.json")
    args = ap.parse_args()

    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    out: dict = {"config": {"requests": args.requests or "full",
                            "splits": args.splits,
                            "reference": "prefix_greedy",
                            "why": ("the paper's §5.2 table mixed frames: its attainable gap was "
                                    "measured from prefix_greedy and its full gap was not. One "
                                    "reference here, and a split sweep so the result cannot be "
                                    "an artifact of where we cut.")},
                 "traces": {}}

    for name, path in TRACES.items():
        p = ROOT / path
        if not p.exists():
            sys.exit(f"missing {path} — see data/README.md")
        wl = load_mooncake(str(p), max_requests=args.requests or None)
        rows = []
        print(f"\n=== {name}: {len(wl.items)} blocks, {len(wl.accesses)} accesses ===")
        print(f"{'split':>6} {'no-pre':>8} {'full gap':>12} {'attainable':>11} "
              f"{'unreachable':>12} {'reachable':>10}")
        for sf in args.splits:
            r = one(wl, sf, sim, w)
            rows.append(r)
            print(f"{sf:>6.1f} {r['no_pre_split_pct']:>7.1f}% {r['full_hindsight_gap']:>12,.0f} "
                  f"{r['attainable_gap']:>11,.0f} {r['unreachable_pct']:>11.1f}% "
                  f"{r['reachable_pct']:>9.1f}%")
        u = [r["unreachable_pct"] for r in rows]
        out["traces"][name] = {"rows": rows,
                              "unreachable_min": min(u), "unreachable_max": max(u)}
        print(f"  unreachable across splits: {min(u):.1f}%–{max(u):.1f}%")

    dest = ROOT / args.out
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
