#!/usr/bin/env python3
"""Open-loop replay client for SGLang (C1-S): as client.py, but no per-request store rule (SGLang's write
policy is a server option) and each request's prompt and cached tokens are read from the final usage chunk
(server started with --enable-cache-report).

Store policies, sent per request in `kv_transfer_params` (each connector reads its own key):
  store-all   nothing sent
  store-none  skip the store of every request
  oracle      skip the store of a request iff it creates no block read again later in the slice
Usage: client_sgl.py WORKLOAD.jsonl OUT.jsonl --url http://127.0.0.1:8100 --model M --speed 1.0 --policy write_through
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp


async def one(session, url, model, row, policy, t_start, out):
    await asyncio.sleep(max(0.0, t_start + row["t_sched"] - time.perf_counter()))
    body = {"model": model, "prompt": row["prompt"], "max_tokens": row["max_tokens"], "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    ttft, n_chunks, err, usage = None, 0, None, None
    try:
        async with session.post(f"{url}/v1/completions", json=body) as resp:
            async for line in resp.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - sent
                n_chunks += 1
                if b'"usage"' in line:
                    u = json.loads(line[6:]).get("usage")
                    if u:
                        usage = u
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    det = (usage or {}).get("prompt_tokens_details") or {}
    out.append({"i": row["i"], "warmup": row["warmup"], "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "cached_tokens": det.get("cached_tokens"),
                "lag_s": round(sent - (t_start + row["t_sched"]), 4),
                "ttft_ms": None if ttft is None else round(ttft * 1e3, 3),
                "e2e_ms": round((time.perf_counter() - sent) * 1e3, 3), "chunks": n_chunks, "err": err})


async def main_async(a):
    rows = [json.loads(l) for l in open(a.workload)]
    for r in rows:
        r["t_sched"] = r["t"] / a.speed
    out: list = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        t_start = time.perf_counter() + 1.0
        await asyncio.gather(*(one(s, a.url, a.model, r, a.policy, t_start, out) for r in rows))
        wall = time.perf_counter() - t_start
    with open(a.out, "w") as f:
        for r in sorted(out, key=lambda r: r["i"]):
            f.write(json.dumps(r) + "\n")
    m = [r for r in out if not r["warmup"] and r["ttft_ms"] is not None]
    t = sorted(r["ttft_ms"] for r in m)
    q = lambda p: t[min(len(t) - 1, int(p * len(t)))] if t else None  # noqa: E731
    print(json.dumps({"policy": a.policy, "speed": a.speed, "measured": len(m),
                      "errors": sum(1 for r in out if r["err"]), "wall_s": round(wall, 1),
                      "ttft_p50_ms": q(0.5), "ttft_p90_ms": q(0.9), "ttft_p99_ms": q(0.99),
                      "ttft_mean_ms": round(sum(t) / len(t), 3) if t else None,
                      "max_send_lag_s": max(r["lag_s"] for r in out),
                      "cached_share": round(sum(r["cached_tokens"] or 0 for r in m)
                                            / max(1, sum(r["prompt_tokens"] or 0 for r in m)), 4)}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--policy", required=True)   # a label: the server's write policy
    asyncio.run(main_async(ap.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
