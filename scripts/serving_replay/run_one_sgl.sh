#!/usr/bin/env bash
# One (write policy, speed) run of SGLang HiCache (notebook §BM; "C1-S" in the probe names). Fresh server per run;
# stopped by its PID.
# Usage: run_one_sgl.sh POLICY SPEED WORKLOAD OUTDIR   POLICY in {none, write_through, write_through_selective, write_back}
set -euo pipefail
POLICY=$1; SPEED=$2; WL=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
SGL_ENV=${SGL_ENV:?set SGL_ENV}
STOA_HF=${STOA_HF:?set STOA_HF}
MODEL=$STOA_HF/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1
GPU=${GPU:-0}; PORT=${PORT:-8100}; CPU_GB=${CPU_GB:-40}; TOKENS=${TOKENS:-73000}
TAG="sgl_${POLICY}_x${SPEED}"
mkdir -p "$OUT"
export HF_HOME=$STOA_HF HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
# FlashInfer JIT-compiles kernels it does not ship for this GPU: it needs ninja (in the env) and nvcc
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME} PATH=$SGL_ENV/bin:$CUDA_HOME/bin:$PATH
ARGS=(-m sglang.launch_server --model-path "$MODEL" --served-model-name m --port "$PORT" --host 127.0.0.1
      --max-total-tokens "$TOKENS" --context-length 24576 --dtype bfloat16 --random-seed 0
      --mem-fraction-static 0.8 --enable-cache-report)
[ "$POLICY" != none ] && ARGS+=(--enable-hierarchical-cache --hicache-size "$CPU_GB" --hicache-write-policy "$POLICY")
# extra server options (e.g. --hicache-io-backend direct), word-split on purpose
[ -n "${SGL_EXTRA:-}" ] && ARGS+=($SGL_EXTRA)
TAG="${TAG}${SGL_TAG:-}"
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d " ")
  [ "$used" -lt 1500 ] && break
  sleep 5
done
"$SGL_ENV/bin/python" "${ARGS[@]}" > "$OUT/$TAG.server.log" 2>&1 & SPID=$!
cleanup() { kill $SPID 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; tail -30 "$OUT/$TAG.server.log"; exit 1; fi
  sleep 2
done
"$SGL_ENV/bin/python" "$HERE/client_sgl.py" "$WL" "$OUT/$TAG.jsonl" --url "http://127.0.0.1:$PORT" --model m \
    --speed "$SPEED" --policy "$POLICY" | tee "$OUT/$TAG.summary.json"
