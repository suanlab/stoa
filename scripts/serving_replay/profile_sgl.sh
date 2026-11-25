#!/usr/bin/env bash
# Why does SGLang's write_through_selective beat write_through at equal hits? nsys over a 60 s window of the
# measured period (120 GB host tier, traceA600 at 0.35x), one run per policy, after the repeat campaigns.
cd "$(dirname "$0")"
until grep -q CAMPAIGN2_DONE campaign2.log 2>/dev/null; do sleep 120; done
export HF_HOME=$HOME/stoa_c1/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-12.8
export PATH=$HOME/stoa_c1/envsgl/bin:$CUDA_HOME/bin:$PATH
MODEL=$HOME/stoa_c1/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1
mkdir -p prof
for p in write_through write_through_selective; do
  while [ "$(free -g | awk 'NR==2{print $7}')" -lt 160 ]; do sleep 60; done
  until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 10; done
  nsys profile -f true -o prof/$p -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --delay 200 --duration 60 \
    $HOME/stoa_c1/envsgl/bin/python -m sglang.launch_server --model-path $MODEL --served-model-name m --port 8100 \
    --host 127.0.0.1 --max-total-tokens 73000 --context-length 24576 --dtype bfloat16 --random-seed 0 \
    --mem-fraction-static 0.8 --enable-cache-report --enable-hierarchical-cache --hicache-size 120 \
    --hicache-write-policy $p > prof/$p.server.log 2>&1 & SPID=$!
  for _ in $(seq 1 300); do curl -sf http://127.0.0.1:8100/v1/models >/dev/null && break; sleep 2; done
  $HOME/stoa_c1/envsgl/bin/python client_sgl.py traceA600.jsonl prof/$p.jsonl --url http://127.0.0.1:8100 --model m \
    --speed 0.35 --policy $p > prof/$p.summary.json
  kill $SPID; for _ in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null
  sleep 20
  nsys stats -q --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum -f csv -o prof/$p prof/$p.nsys-rep \
    > prof/$p.stats.log 2>&1
done
echo PROFILE_DONE
