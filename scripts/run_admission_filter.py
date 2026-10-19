#!/usr/bin/env python3
"""H2: does an admission filter for unseen blocks help an online cache on these traces?

76-78% of Mooncake blocks are touched once, the regime in which Akamai caches on the SECOND request
(Maggs & Sitaraman) and TinyLFU admits a newcomer only if it is more frequent than the victim
(Einziger et al., TOS 2017). Online LRU cache over the full trace, hit rate as a share of Belady's
(demand-paging Belady, which always admits: an admission filter can in principle exceed it, so LRU's
own hit rate is reported alongside):
  lru            always admit (must equal simulate_fast's LRU exactly: asserted)
  second-hit     admit a block on its second access ever (exact doorkeeper set)
  tinylfu        admit on a miss only if the newcomer's frequency exceeds the LRU victim's; counts
                 are halved every 10 x cache-size accesses (TinyLFU's reset), exact counts
H2 predicts second-hit beats lru by >= 5 points of Belady at the 2% tier on both production traces,
and the gain shrinks or reverses as the tier grows (falsified if no gain at 2%).
Usage: python3 scripts/run_admission_filter.py
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arrival_admission import TRACES  # noqa: E402
from stoa.eval.online import simulate_fast  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

FRACS = (0.02, 0.05, 0.10, 0.20)


def run(accesses, slots, policy):
    cache: OrderedDict[str, None] = OrderedDict()
    seen: set[str] = set()
    freq: dict[str, float] = defaultdict(float)
    reset_every, n = 10 * slots, 0
    hits = 0
    for _t, b in accesses:
        n += 1
        freq[b] += 1
        if policy == "tinylfu" and n % reset_every == 0:
            for k in list(freq):
                freq[k] /= 2
        if b in cache:
            hits += 1
            cache.move_to_end(b)
            continue
        admit = True
        if policy == "second-hit":
            admit = b in seen
            seen.add(b)
        elif policy == "tinylfu" and len(cache) >= slots:
            victim = next(iter(cache))
            admit = freq[b] > freq[victim]
        if not admit:
            continue
        if len(cache) >= slots:
            cache.popitem(last=False)
        cache[b] = None
    return hits / len(accesses)


def main() -> int:
    rows = []
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        for frac in FRACS:
            slots = max(1, int(frac * len(wl.items)))
            bel = simulate_fast(wl, "belady", slots).hit_rate
            ref_lru = simulate_fast(wl, "lru", slots).hit_rate
            r = {p: run(wl.accesses, slots, p) for p in ("lru", "second-hit", "tinylfu")}
            assert abs(r["lru"] - ref_lru) < 1e-12, f"LRU mismatch {r['lru']} vs {ref_lru}"
            rows.append({"trace": name, "cache_frac": frac, "belady": round(bel, 4),
                         **{f"{p}_over_belady": round(v / bel, 4) for p, v in r.items()},
                         **{p: round(v, 4) for p, v in r.items()}})
            print(rows[-1])
    (ROOT / "experiments" / "admission_filter.json").write_text(json.dumps(
        {"config": {"fracs": list(FRACS), "belady": "demand paging (always admits)",
                    "tinylfu_reset": "halve counts every 10 x slots accesses"}, "rows": rows}, indent=2) + "\n")
    print("wrote experiments/admission_filter.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
