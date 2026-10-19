#!/usr/bin/env python3
"""How the split-instant reachable share depends on trace length (notebook AS).

Under the corrected bill (a new block's creation is a compulsory miss) the share of the hindsight gap a
split-time policy can reach is large on short prefixes and small on full traces: the protocol defect is
a long-horizon effect. Same computation as run_reachable_ceiling.py at split 0.5, on the first N requests.
Usage: python3 scripts/run_reachable_by_length.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import attainable_ceiling, place_with_beliefs, prefix_greedy_cost  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
LENGTHS = (1500, 3000, 6000, None)


def main() -> int:
    rows = []
    for name, path in TRACES.items():
        for n in LENGTHS:
            wl = load_mooncake(str(ROOT / path), max_requests=n)
            st = max(1, wl.time_split(0.5))
            stats = wl.prefix_stats(st)
            visible = [i.item_id for i in wl.items if stats[i.item_id]["count"] > 0]
            fut = wl.future_counts(st)
            c_pref = prefix_greedy_cost(wl)
            c_orac = place_with_beliefs(wl, {i.item_id: float(fut.get(i.item_id, 0)) for i in wl.items})
            reach = 100.0 * (c_pref - attainable_ceiling(wl, visible)) / (c_pref - c_orac)
            rows.append({"trace": name, "requests": n or "full", "reachable_pct": round(reach, 1)})
            print(rows[-1], flush=True)
    (ROOT / "experiments" / "reachable_by_length.json").write_text(
        json.dumps({"config": {"split": 0.5, "bill": "creation accesses are compulsory misses"},
                    "rows": rows}, indent=2) + "\n")
    print("wrote experiments/reachable_by_length.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
