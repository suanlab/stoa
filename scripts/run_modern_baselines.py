#!/usr/bin/env python3
"""Modern one-hit-wonder-aware baselines on the production KV traces (§BB).

Same traces, tier sizes and hit-rate definition as run_reactive_real.py (§V-F), adding S3-FIFO, SIEVE
and W-TinyLFU (stoa.modern) beside Belady, LRU and ARC recomputed here, so every ratio in a row shares
one Belady. Reported as a fraction of Belady's hit rate.
Usage: python3 scripts/run_modern_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stoa.eval.online import simulate_arc, simulate_fast  # noqa: E402
from stoa.modern import simulate_s3fifo, simulate_sieve, simulate_wtinylfu  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
FRACS = (0.02, 0.05, 0.10, 0.20)


def main() -> int:
    out = {"config": {"cache_fracs": list(FRACS), "metric": "hit rate / Belady hit rate",
                      "slots": "max(1, int(frac * blocks)), as run_reactive_real.py"}, "traces": {}}
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        rows = []
        for frac in FRACS:
            slots = max(1, int(frac * len(wl.items)))
            hr = {"belady": simulate_fast(wl, "belady", slots).hit_rate,
                  "lru": simulate_fast(wl, "lru", slots).hit_rate,
                  "arc": simulate_arc(wl, slots).hit_rate,
                  "s3fifo": simulate_s3fifo(wl, slots).hit_rate,
                  "sieve": simulate_sieve(wl, slots).hit_rate,
                  "wtinylfu": simulate_wtinylfu(wl, slots).hit_rate}
            row = {"cache_frac": frac, **{k: round(v, 4) for k, v in hr.items()},
                   **{f"{k}_over_belady": round(v / hr["belady"], 4) for k, v in hr.items() if k != "belady"}}
            rows.append(row)
            print(name, frac, {k: row[f"{k}_over_belady"] for k in hr if k != "belady"}, flush=True)
        out["traces"][name] = {"rows": rows}
    (ROOT / "experiments" / "modern_baselines.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/modern_baselines.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
