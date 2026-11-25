#!/usr/bin/env bash
# Repeats in a seeded shuffled order (campaign_runs.txt: CPU_GB CONNECTOR POLICY SPEED WORKLOAD per line);
# each run in its own directory so repeats of a cell never overwrite each other.
cd "$(dirname "$0")"
export STOA_ENV=$HOME/stoa_c1/env STOA_HF=$HOME/stoa_c1/hf GPU=1 GPU_UTIL=0.53
n=0
while read -r cap c p s wl; do
  n=$((n+1))
  [ -z "$cap" ] && continue
  while [ "$(free -g | awk 'NR==2{print $7}')" -lt $((cap + 40)) ]; do sleep 60; done
  if [ "$wl" = traceA600.jsonl ]; then export STOA_WARMUP_REQS=200; else export STOA_WARMUP_REQS=400; fi
  CPU_GB=$cap ./run_one.sh "$c" "$p" "$s" "$wl" "campaign/$(printf %02d $n)_cpu$cap"
done < campaign_runs.txt
echo CAMPAIGN_DONE
