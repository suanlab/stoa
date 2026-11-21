#!/usr/bin/env bash
# One (connector, policy, speed) run of the serving replay (notebook §BM). Starts a fresh vLLM server (so no
# cache state is shared between runs), replays the workload, then stops the server by its PID.
# Usage: run_one.sh CONNECTOR POLICY SPEED WORKLOAD OUTDIR      CONNECTOR in {none, lmcache, offload}
set -euo pipefail
CONN=$1; POLICY=$2; SPEED=$3; WL=$4; OUT=$5
HERE=$(cd "$(dirname "$0")" && pwd)
# host-specific: STOA_ENV is a Python env with vLLM 0.11.0 + LMCache 0.3.9, STOA_HF an HF_HOME holding the model
STOA_ENV=${STOA_ENV:?set STOA_ENV}
STOA_HF=${STOA_HF:?set STOA_HF}
PY=$STOA_ENV/bin/python
MODEL=$STOA_HF/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1
GPU=${GPU:-0}
PORT=${PORT:-8100}
GPU_UTIL=${GPU_UTIL:-0.30}
CPU_GB=${CPU_GB:-40}
TAG="${CONN}_${POLICY}_x${SPEED}"
mkdir -p "$OUT"
export HF_HOME=$STOA_HF HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH}
ARGS=(serve "$MODEL" --served-model-name m --port "$PORT" --gpu-memory-utilization "$GPU_UTIL"
      --max-model-len 24576 --dtype bfloat16 --seed 0 --disable-log-requests)
case $CONN in
  none) ;;
  lmcache)
    export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=$CPU_GB LMCACHE_SAVE_UNFULL_CHUNK=True
    ARGS+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}') ;;
  offload)
    # 2 MiB per 16-token block for this model (32 layers x 8 KV heads x 128 x 2 x bf16 x 16 tokens)
    NB=$(( CPU_GB * 1024 / 2 ))
    ARGS+=(--kv-transfer-config "{\"kv_connector\":\"SkipStoreOffloadingConnector\",\"kv_connector_module_path\":\"skipstore_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":$NB}}") ;;
  *) echo "unknown connector $CONN"; exit 2 ;;
esac
# wait until the GPU holds no other memory (a just-stopped process can take a while to release it)
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d " ")
  [ "$used" -lt 1500 ] && break
  sleep 5
done
start_server() { "$STOA_ENV/bin/vllm" "${ARGS[@]}" > "$OUT/$TAG.server.log" 2>&1 & SPID=$!; }
start_server
cleanup() { kill $SPID 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then
    if [ -z "${RETRIED:-}" ]; then RETRIED=1; echo "server died once; retrying"; cp "$OUT/$TAG.server.log" "$OUT/$TAG.server.fail1.log"; sleep 20; start_server; continue; fi
    echo "server died"; tail -30 "$OUT/$TAG.server.log"; exit 1
  fi
  sleep 2
done
$PY "$HERE/client.py" "$WL" "$OUT/$TAG.jsonl" --url "http://127.0.0.1:$PORT" --model m --speed "$SPEED" \
    --policy "$POLICY" | tee "$OUT/$TAG.summary.json"
