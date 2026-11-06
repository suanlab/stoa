#!/usr/bin/env python3
"""Where post-split reuse falls: dead, live and new blocks (§BA).

For each split, the share of visible blocks never read again, and the share of billable post-split
accesses (creations excluded, as everywhere) that fall on visible blocks versus new ones.
Usage: python3 scripts/run_reuse_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_arrival_admission import TRACES  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

SPLITS = (0.3, 0.4, 0.5, 0.6, 0.7)


def main() -> int:
    rows = []
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        for split in SPLITS:
            st = max(1, wl.time_split(split))
            fut = wl.future_counts(st)
            vis = {i.item_id for i in wl.items if wl.traces[i.item_id].access_steps[0] < st}
            live = {b for b in vis if fut.get(b, 0) > 0}
            tot = sum(fut.values())
            rows.append({"trace": name, "split": split, "n_visible": len(vis), "n_live_visible": len(live),
                         "dead_visible_pct": round(100 * (len(vis) - len(live)) / len(vis), 2),
                         "reuse_on_new_pct": round(100 * sum(v for b, v in fut.items() if b not in vis) / tot, 2),
                         "reuse_on_live_visible_pct": round(100 * sum(fut[b] for b in live) / tot, 2)})
            print(rows[-1], flush=True)
    (ROOT / "experiments" / "reuse_split.json").write_text(json.dumps(
        {"config": {"splits": list(SPLITS), "reuse": "billable post-split accesses (creations excluded)"},
         "rows": rows}, indent=2) + "\n")
    print("wrote experiments/reuse_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
