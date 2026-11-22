#!/usr/bin/env python3
"""Pre-registration aid for the SGLang HiCache runs: host/GPU hits of the three write policies on a workload.

Two LRU tiers at 256-token chunk granularity, a request matching its prefix chunk by chunk (GPU, else host)
until the first miss:
  write_through            every chunk is written to host when created (host holds a copy of everything)
  write_through_selective  a chunk is written to host on its second access, if it is still on the GPU then
  write_back               a chunk is written to host when the GPU evicts it
Usage: sim_hicache.py WORKLOAD.jsonl GPU_GIB HOST_GIB [HOST_GIB ...]
"""
import hashlib
import json
import sys
from collections import OrderedDict

CH, KB = 256, 131072
wl = [json.loads(line) for line in open(sys.argv[1])]
gpu_cap = int(float(sys.argv[2]) * 2**30 / (CH * KB))
chains = []
for r in wl:
    p, h, c = r["prompt"], b"", []
    for k in range((len(p) + CH - 1) // CH):
        h = hashlib.blake2b(h + json.dumps(p[k * CH:(k + 1) * CH]).encode(), digest_size=16).digest()
        c.append(h)
    chains.append(c)


def run(policy, host_cap):
    gpu, host, hits = OrderedDict(), OrderedDict(), {}
    g_hit = h_hit = tot = writes = 0

    def host_put(h):
        nonlocal writes
        writes += 1
        host[h] = 1
        host.move_to_end(h)
        while len(host) > host_cap:
            host.popitem(last=False)

    def gpu_put(h):
        gpu[h] = 1
        gpu.move_to_end(h)
        while len(gpu) > gpu_cap:
            old, _ = gpu.popitem(last=False)
            if policy == "write_back" and old not in host:
                host_put(old)
            elif old not in host:
                hits.pop(old, None)                   # SGLang deletes the node: its hit count goes with it

    for r, c in zip(wl, chains):
        n_g = n_h = 0
        for h in c:
            if h in gpu:
                n_g += 1
            elif h in host:
                n_h += 1
                host.move_to_end(h)
            else:
                break
        if not r["warmup"]:
            g_hit, h_hit, tot = g_hit + n_g, h_hit + n_h, tot + len(c)
        for h in c:                                   # every chunk of the request ends up on the GPU
            hits[h] = hits.get(h, 0) + 1
            if h in host:
                host.move_to_end(h)
            elif policy == "write_through" or (policy == "write_through_selective" and hits[h] >= 2):
                host_put(h)
            gpu_put(h)
    return {"gpu_hit": round(g_hit / tot, 4), "host_hit": round(h_hit / tot, 4), "host_writes": writes}


for cap in sys.argv[3:]:
    hc = int(float(cap) * 2**30 / (CH * KB))
    for pol in ("write_through", "write_through_selective", "write_back"):
        print(cap, pol, run(pol, hc))
print("gpu only", run("none", 0))
