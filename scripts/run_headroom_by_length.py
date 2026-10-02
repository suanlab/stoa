#!/usr/bin/env python3
"""Headroom (heuristic vs. hindsight oracle) as a function of trace length, from code.

Why this script exists. Two figures in the paper -- Table 2's "headroom, heuristic vs. oracle
87%" and §5.8's free-parameter example "a headroom taken at 6% of the trace moved six points by
full length (81.0% to 87.2%)" -- came from `mooncake_length_sweep.json`, which has no producer
script and, on inspection, no headroom column at all: it stores AUC and captured fraction only.
The numbers were computed in-session before the §W tie-break fix and written into the paper
by hand. Meanwhile §5.8's calibration paragraph states the same quantity at full length as
80%, so the paper carried two values for one number.

This recomputes it through `reference_costs` -- the routine every other headroom in the paper
uses -- at each length, and records the trace sizes the paper states, so the length grid has a
producer and the paper has one value.

    python3 scripts/run_headroom_by_length.py
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
from stoa.sequential import reference_costs  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
LENGTHS = (1500, 6000, 12031, 23608)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="experiments/headroom_by_length.json")
    args = ap.parse_args()

    rows = []
    print(f"{'trace':<13} {'requests':>9} {'full':>5} {'blocks':>8} {'headroom':>9}")
    for name, path in TRACES.items():
        p = ROOT / path
        if not p.exists():
            sys.exit(f"missing {path} — see data/README.md")
        total = sum(1 for _ in p.open())
        for n in LENGTHS:
            if n > total:
                continue                       # this trace is exhausted before n
            wl = load_mooncake(str(p), max_requests=n)
            ref = reference_costs(wl)
            r = {"trace": name, "requests": n, "full_trace": n == total,
                 "n_blocks": len(wl.items),
                 "oracle_cost": round(ref["oracle_future"], 1),
                 "heuristic_cost": round(ref["prefix_greedy"], 1),
                 "headroom_pct": round(ref["headroom_pct"], 2)}
            rows.append(r)
            print(f"{name:<13} {n:>9,} {str(r['full_trace']):>5} {r['n_blocks']:>8,} "
                  f"{r['headroom_pct']:>8.2f}%")
        if not any(r["trace"] == name and r["full_trace"] for r in rows):
            sys.exit(f"{name}: no length equals the full trace ({total}); add it to LENGTHS")

    out = {"config": {"lengths": list(LENGTHS),
                      "headroom": "(prefix_greedy - oracle) / prefix_greedy, via reference_costs",
                      "note": "conversation exhausts at 12,031 requests; toolagent at 23,608"},
           "rows": rows}
    (ROOT / args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
