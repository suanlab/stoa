#!/usr/bin/env python3
"""C1 workload: a slice of a production prefix trace as token-id prompts with the trace's prefix structure.

Each hash id becomes a fixed pseudo-random run of `block_tokens` token ids (seeded by the id), so two requests
that share a hash prefix share a token prefix and hit the same KV blocks in vLLM / LMCache / the offloading
connector. Read-only on the repository's data; writes one JSONL file.

Per request: t (seconds from the slice start), prompt (token ids), max_tokens (trace output length, capped),
warmup (the first `--warmup` requests are replayed but not measured), store_oracle (True iff one of the
blocks this request creates -- first appearance within the slice -- is read again later in the slice),
new_tokens (tokens of blocks it creates).
Usage: make_workload.py TRACE.jsonl OUT.jsonl --block-tokens 16 --start 10000 --n 3000 --warmup 1000
"""
from __future__ import annotations

import argparse
import json
import random

VOCAB_LO, VOCAB_HI = 1000, 120000          # inside Llama-3's 128,256-token vocabulary, clear of special ids


def chunk(h: int, n: int) -> list[int]:
    r = random.Random(h * 2654435761 % 2**32)
    return [r.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--block-tokens", type=int, required=True)
    ap.add_argument("--start", type=int, default=10000)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-out", type=int, default=64)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.trace)]
    rows.sort(key=lambda r: r["timestamp"])
    sl = rows[a.start:a.start + a.n]
    t0 = sl[0]["timestamp"]
    seen, created = set(), []
    for r in sl:
        c = [h for h in r["hash_ids"] if h not in seen]
        seen.update(c)
        created.append(c)
    # last access index of every block in the slice
    last = {}
    for i, r in enumerate(sl):
        for h in r["hash_ids"]:
            last[h] = i
    cache: dict[int, list[int]] = {}
    with open(a.out, "w") as f:
        for i, r in enumerate(sl):
            prompt = []
            for h in r["hash_ids"]:
                if h not in cache:
                    cache[h] = chunk(h, a.block_tokens)
                prompt += cache[h]
            f.write(json.dumps({
                "i": i, "t": round(r["timestamp"] - t0, 4), "prompt": prompt,
                "max_tokens": max(1, min(int(r.get("output_length", 1)), a.max_out)),
                "warmup": i < a.warmup,
                "store_oracle": any(last[h] > i for h in created[i]),
                "new_tokens": len(created[i]) * a.block_tokens}) + "\n")
    n_meas = a.n - a.warmup
    meas = [created[i] for i in range(a.warmup, a.n)]
    print(json.dumps({"requests": a.n, "measured": n_meas, "span_s": round(sl[-1]["timestamp"] - t0, 1),
                      "mean_prompt_tokens": round(sum(len(r["hash_ids"]) for r in sl) * a.block_tokens / a.n),
                      "oracle_store_share_measured": round(sum(
                          any(last[h] > i for h in created[i]) for i in range(a.warmup, a.n)) / n_meas, 4),
                      "created_tokens_measured": sum(len(c) for c in meas) * a.block_tokens}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
