#!/usr/bin/env python3
"""Reactive tiering on the agentic source (kv-cache-tester): the tier-size sweep.

`experiments/reactive_kvct_sweep.json` was cited by the paper (the eviction-signal inversion in
§V-F) and by REPRODUCIBILITY.md, but no script in the repository produced it: the claim table said
"`simulate_fast` sweep over `cache_frac`". Found on the 2026-10-19 re-verification pass; this is
that sweep, written down, and it reproduces the stored artifact exactly.

Samples are the first N conversation files (the loader's default), pooled and interleaved by
wall-clock time. That is a prefix sample, which is why the paper reports these as a range and
never as a point estimate (§V-E).

CPU, a few minutes. Needs the kv-cache-tester download (data/README.md).
Usage: python3 scripts/run_reactive_kvct.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stoa.eval.online import simulate_fast  # noqa: E402
from stoa.kvct import load_kvct  # noqa: E402

CONVS = (20, 40)
FRACS = (0.01, 0.02, 0.05, 0.10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="experiments/reactive_kvct_sweep.json")
    args = ap.parse_args()
    if not (ROOT / "data" / "kvct").exists():
        sys.exit("missing data/kvct; see data/README.md (kv-cache-tester)")
    rows = []
    for n in CONVS:
        wl = load_kvct(str(ROOT / "data" / "kvct"), limit=n)
        for frac in FRACS:
            slots = max(1, int(frac * len(wl.items)))
            r = {k: simulate_fast(wl, k, slots).hit_rate for k in ("belady", "lru", "h2o", "static")}
            b = r["belady"]
            best = max(r["lru"], r["h2o"], r["static"])
            rows.append({"convs": n, "cache_frac": frac, "belady": round(b, 4),
                         "lru": round(r["lru"], 4), "h2o": round(r["h2o"], 4),
                         "static": round(r["static"], 4),
                         "best_reactive_over_belady": round(best / b, 4) if b > 0 else 0.0})
            print(rows[-1])
    out = ROOT / args.out
    out.write_text(json.dumps({"source": "kv-cache-tester", "rows": rows}) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
