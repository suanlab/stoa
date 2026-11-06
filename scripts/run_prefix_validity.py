#!/usr/bin/env python3
"""Is a block hit a usable hit? Per-block versus prefix-usable hit rates (§BD; review E4).

Every replay in the paper bills blocks independently. A KV block is usable only behind its prefix: a
serving engine reuses blocks 0..j of a request's hash chain only if all of them are cached, so a cached
block j after a missed block i < j is recomputed anyway. This script replays each trace request by
request, blocks in their prefix order, through a single-tier LRU and Belady cache at the paper's four tier
sizes, and counts a hit two ways:

  block    the block is resident (the paper's model)
  prefix   the block is resident and so is every earlier block of the same request

The gap between the two is what the independent-block model overstates. Uniform blocks; a block's first
access is a compulsory miss in both counts.
Usage: python3 scripts/run_prefix_validity.py
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = {"conversation": "data/mooncake_conversation_trace.jsonl",
          "toolagent": "data/mooncake_toolagent_trace.jsonl"}
FRACS = (0.02, 0.05, 0.10, 0.20)


def requests(path):
    with open(ROOT / path) as fh:
        return [json.loads(line).get("hash_ids", []) for line in fh]


def replay(reqs, slots, policy):
    """Returns (block hits, prefix-usable hits, accesses)."""
    flat = [h for r in reqs for h in r]
    nxt = [0] * len(flat)
    if policy == "belady":
        last = {}
        for i in range(len(flat) - 1, -1, -1):
            nxt[i] = last.get(flat[i], float("inf"))
            last[flat[i]] = i
    cache: OrderedDict = OrderedDict()            # LRU order, or key -> next use for Belady
    import heapq
    heap: list = []                               # Belady: (-next_use, key), lazily invalidated
    bh = ph = 0
    i = 0
    for r in reqs:
        prefix_ok = True
        for h in r:
            hit = h in cache
            bh += hit
            prefix_ok = prefix_ok and hit
            ph += prefix_ok
            if policy == "lru":
                if hit:
                    cache.move_to_end(h)
                else:
                    cache[h] = None
                    if len(cache) > slots:
                        cache.popitem(last=False)
            else:
                cache[h] = nxt[i]
                heapq.heappush(heap, (-nxt[i], h))
                while len(cache) > slots:
                    nu, k = heapq.heappop(heap)
                    if k in cache and cache[k] == -nu:
                        del cache[k]
            i += 1
    return bh, ph, len(flat)


def main() -> int:
    out = {"config": {"cache_fracs": list(FRACS), "order": "requests in file order, blocks in hash-chain order",
                      "slots": "max(1, int(frac * distinct blocks))"}, "traces": {}}
    for name, path in TRACES.items():
        reqs = requests(path)
        n_blocks = len({h for r in reqs for h in r})
        # The structural reason the two counts can agree: a block's id fixes its predecessor (hash
        # chain), so every access to block j is also an access to its whole prefix. Counted, not assumed.
        pred, other_pred = {}, 0
        for r in reqs:
            for k, h in enumerate(r):
                p = r[k - 1] if k else None
                other_pred += h in pred and pred[h] != p
                pred.setdefault(h, p)
        rows = []
        for frac in FRACS:
            slots = max(1, int(frac * n_blocks))
            row = {"cache_frac": frac}
            for pol in ("lru", "belady"):
                bh, ph, n = replay(reqs, slots, pol)
                row[pol] = {"block_hit_rate": round(bh / n, 4), "prefix_hit_rate": round(ph / n, 4),
                            "unusable_share_of_hits": round(1 - ph / bh, 4) if bh else 0.0}
            rows.append(row)
            print(name, frac, row, flush=True)
        out["traces"][name] = {"n_blocks": n_blocks, "n_positions": sum(map(len, reqs)),
                               "positions_with_another_predecessor": other_pred, "rows": rows}
    (ROOT / "experiments" / "prefix_validity.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote experiments/prefix_validity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
