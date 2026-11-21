#!/usr/bin/env bash
# The serving-replay matrix of notebook §BM, as run on one RTX A6000 48 GB (PCIe Gen3 x16), GPU_UTIL=0.53.
#   stage 1: traceA1200 (1,200 requests, 400 warm-up), host tier 40 GB; at 0.35x and 0.2x of the trace's
#            arrival rate no connector, then LMCache and the offloading connector x store-all / oracle /
#            store-none; at 0.1x store-all and oracle on both connectors.
#   stage 2: traceA600 (600 requests, 200 warm-up), host tier 120 GB (where store-all and oracle reach the
#            same host hit ratio) and 40 GB; both connectors x store-all / oracle at 0.35x and 0.2x.
# Every run starts a fresh server, so no cache state crosses runs. Before each run the host must have the
# tier plus 40 GB of memory available (the tier is pinned host memory on a shared machine).
# Usage: STOA_ENV=... STOA_HF=... [GPU=0] matrix.sh {1|2} OUTDIR   (workloads: make_workload.py, below)
#   make_workload.py data/bailian_qwen_traceA_blksz_16.jsonl traceA1200.jsonl --block-tokens 16 --start 10000 --n 1200 --warmup 400
#   make_workload.py data/bailian_qwen_traceA_blksz_16.jsonl traceA600.jsonl  --block-tokens 16 --start 10000 --n 600 --warmup 200
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=$1; OUT=$2
export GPU_UTIL=${GPU_UTIL:-0.53}
run() {  # CPU_GB CONNECTOR POLICY SPEED WORKLOAD DIR
  while [ "$(free -g | awk 'NR==2{print $7}')" -lt $(( $1 + 40 )) ]; do sleep 60; done
  CPU_GB=$1 "$HERE/run_one.sh" "$2" "$3" "$4" "$5" "$6"
}
if [ "$STAGE" = 1 ]; then
  for s in 0.35 0.2; do
    run 40 none store-all $s traceA1200.jsonl "$OUT/stage1_cpu40"
    for c in lmcache offload; do for p in store-all oracle store-none; do
      run 40 $c $p $s traceA1200.jsonl "$OUT/stage1_cpu40"; done; done
  done
  for c in lmcache offload; do for p in store-all oracle; do
    run 40 $c $p 0.1 traceA1200.jsonl "$OUT/stage1_cpu40"; done; done
else
  export STOA_WARMUP_REQS=200
  for cap in 120 40; do for s in 0.35 0.2; do for c in lmcache offload; do for p in store-all oracle; do
    run $cap $c $p $s traceA600.jsonl "$OUT/stage2_cpu$cap"; done; done; done; done
fi
