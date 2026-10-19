#!/usr/bin/env python3
"""Descriptive statistics of both sources: the numbers in Table 2.

Table 2 stated request counts, one-time fractions, mean reuse and blocks per request for both
sources with no producing script, and its agentic-source headroom (81--94%) came from the
nested-prefix placement runs §5.5 retracts. Found on the 2026-10-19 re-verification pass. The
headroom row now cites the randomized sampling study (`run_sampling_study.py`); every other cell
comes from here.

Mooncake rows are per trace. kv-cache-tester rows are per sample size (the first N conversation
files), because its descriptive statistics drift with N -- the reason Table 2 says so.

CPU, about ten minutes (the 200-conversation load dominates). Needs the Mooncake traces and the
kv-cache-tester download (data/README.md). Usage: python3 scripts/run_source_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stoa.kvct import load_kvct  # noqa: E402
from stoa.mooncake import load_mooncake, trace_summary  # noqa: E402

MOONCAKE = {"conversation": "data/mooncake_conversation_trace.jsonl",
            "toolagent": "data/mooncake_toolagent_trace.jsonl"}
KVCT_SIZES = (20, 40, 80, 120, 150, 200)


def _row(wl) -> dict:
    s = trace_summary(wl)
    requests = len({t for t, _ in wl.accesses})
    return {"requests": requests, "blocks": s["items"], "accesses": s["accesses"],
            "one_time_frac": s["one_time_frac"], "mean_reuse": s["mean_reuse"],
            "blocks_per_request": round(s["accesses"] / requests, 1)}


def main() -> int:
    out = {"mooncake": {}, "kvct": {}}
    for name, path in MOONCAKE.items():
        if not (ROOT / path).exists():
            sys.exit(f"missing {path}; see data/README.md")
        out["mooncake"][name] = _row(load_mooncake(str(ROOT / path)))
        print(name, out["mooncake"][name])
    if not (ROOT / "data" / "kvct").exists():
        sys.exit("missing data/kvct; see data/README.md (kv-cache-tester)")
    for n in KVCT_SIZES:
        out["kvct"][str(n)] = _row(load_kvct(str(ROOT / "data" / "kvct"), limit=n))
        print(f"kvct {n}", out["kvct"][str(n)])
    dest = ROOT / "experiments" / "source_stats.json"
    dest.write_text(json.dumps({"config": {"kvct_sample": "first N conversation files"},
                                **out}, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
