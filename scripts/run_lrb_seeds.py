#!/usr/bin/env python3
"""Does "LRB below ARC at all eight operating points" survive LRB's own randomness? (§BJ)

LRB samples eviction candidates and training rows with a seeded RNG; every LRB number in the paper is
seed 0. A review (2026-11-09, R4) noted that the narrowest margin is 0.25 points of Belady (conversation,
2% tier: LRB 29.18 under the adopted repair, ARC 29.43) and asked for seed repeats where it matters.
This script reruns both repairs of the truncation bug (`all_history`, `sliding_window`) with seeds 0-9
at the two tightest operating points (2% tier, both traces), where the margins are smallest; ARC and
Belady are deterministic and computed once, through the same functions as run_lrb_retraction.py.

Registered before running: the claim "LRB below ARC at all eight operating points" is kept if every seed
of both repairs is below ARC at both points; otherwise the paper states the claim for seed 0 only and
reports how many seeds cross.
Usage: python3 scripts/run_lrb_seeds.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
FRAC = 0.02
SEEDS = tuple(range(10))
POLICIES = ("all_history", "sliding_window")


def _lrb(job):
    trace, policy, seed = job
    from stoa.lrb import simulate_lrb
    from stoa.mooncake import load_mooncake
    wl = load_mooncake(str(ROOT / TRACES[trace]))
    slots = max(1, int(FRAC * len(wl.items)))
    return trace, policy, seed, simulate_lrb(wl, slots, buffer_policy=policy, seed=seed).hit_rate


def _ref(trace):
    from stoa.eval.online import simulate_arc, simulate_fast
    from stoa.mooncake import load_mooncake
    wl = load_mooncake(str(ROOT / TRACES[trace]))
    slots = max(1, int(FRAC * len(wl.items)))
    return trace, simulate_fast(wl, "belady", slots).hit_rate, simulate_arc(wl, slots).hit_rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()
    jobs = [(t, p, s) for t in TRACES for p in POLICIES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        refs = {t: (b, a) for t, b, a in ex.map(_ref, TRACES)}
        res = list(ex.map(_lrb, jobs))
    rows = []
    for t in TRACES:
        b, a = refs[t]
        for p in POLICIES:
            vals = [round(h / b, 4) for tt, pp, _s, h in sorted(res, key=lambda r: r[2]) if tt == t and pp == p]
            rows.append({"trace": t, "cache_frac": FRAC, "repair": p, "arc": round(a / b, 4),
                         "lrb_by_seed": vals, "min": min(vals), "max": max(vals),
                         "seeds_at_or_above_arc": sum(v >= round(a / b, 4) for v in vals)})
            print(rows[-1], flush=True)
    (ROOT / "experiments" / "lrb_seeds.json").write_text(json.dumps(
        {"config": {"cache_frac": FRAC, "seeds": list(SEEDS), "repairs": list(POLICIES),
                    "normalisation": "hit rate / Belady hit rate",
                    "registered": "claim kept if no seed of either repair reaches ARC at either point"},
         "rows": rows}, indent=2) + "\n")
    print("wrote experiments/lrb_seeds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
