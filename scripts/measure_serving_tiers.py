#!/usr/bin/env python3
"""Time to first token by KV tier in a real serving stack: vLLM + LMCache (notebook §BE).

The paper's cost model prices a tier read from a declared latency table, and §AX timed raw block copies.
This script asks the serving engine itself: for a shared prefix of P tokens followed by a short unique
suffix, how long does the first token take when the prefix's KV is

  recompute   not cached anywhere (the prefix is prefilled)
  gpu         in vLLM's GPU prefix cache
  cpu         evicted from the GPU, held by LMCache in host memory
  disk        evicted from the GPU, held by LMCache on local disk (page cache dropped before the read)

and what does storing it cost the request that creates it (`store_overhead`: a cold request with
LMCache saving minus one without). One mode per process, since LMCache is configured from the
environment before vLLM starts:

  python3 measure_serving_tiers.py --model <path> --mode {none,cpu,disk} --out <json>
    none : vLLM alone, prefix caching on -> recompute and gpu
    cpu  : LMCache local CPU backend      -> cold-with-store and cpu
    disk : LMCache local disk backend     -> cold-with-store and disk  (--disk-dir on the disk tier)

Prompts are token ids drawn at random per repetition, so every prefix is new to every cache and lengths
are exact. Wall time of `generate(max_tokens=1)` is the measured quantity; medians over --reps.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

LENGTHS = (512, 2048, 8192)
SUFFIX = 16


def fadvise_dir(d: Path) -> int:
    """Evict a directory's files from the page cache (no root needed for clean pages)."""
    n = 0
    for p in d.rglob("*"):
        if p.is_file():
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("none", "cpu", "disk"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--disk-dir", default="")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    # LMCache stages disk I/O through a host buffer; one larger than a prefix can serve "disk" reads from
    # memory. The disk mode is reported at the buffer size it ran with.
    ap.add_argument("--disk-cpu-buffer-gb", type=float, default=4.0)
    # Cold disk: store every prefix first, then evict the page cache AND stream --flush-gb of unrelated data
    # from --flush-dir (same array) to push the controller's cache out, then load each prefix once.
    ap.add_argument("--disk-cold", action="store_true")
    ap.add_argument("--flush-dir", default="")
    ap.add_argument("--flush-gb", type=float, default=16.0)
    a = ap.parse_args()

    if a.mode in ("cpu", "disk"):
        os.environ["LMCACHE_CHUNK_SIZE"] = "256"
        os.environ["LMCACHE_SAVE_UNFULL_CHUNK"] = "True"
        if a.mode == "cpu":
            os.environ["LMCACHE_LOCAL_CPU"] = "True"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20"
        else:
            Path(a.disk_dir).mkdir(parents=True, exist_ok=True)
            os.environ["LMCACHE_LOCAL_CPU"] = "False"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(a.disk_cpu_buffer_gb)
            os.environ["LMCACHE_LOCAL_DISK"] = f"file://{Path(a.disk_dir).resolve()}/"
            os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "40"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = dict(model=a.model, enable_prefix_caching=True, gpu_memory_utilization=a.gpu_mem,
              max_model_len=max(LENGTHS) + 512, dtype="bfloat16", seed=0)
    if a.mode != "none":
        from vllm.config import KVTransferConfig
        kw["kv_transfer_config"] = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)

    def toks(n):
        return [rng.randrange(1000, vocab - 1000) for _ in range(n)]

    def ttft(ids):
        t0 = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return (time.perf_counter() - t0) * 1e3

    for _ in range(3):                                   # warm up kernels and the connector
        ttft(toks(512) + toks(SUFFIX))
    llm.reset_prefix_cache()

    res = {}
    if a.mode == "disk" and a.disk_cold:
        stored = {P: [] for P in LENGTHS}
        cold = {P: [] for P in LENGTHS}
        for P in LENGTHS:
            for _ in range(a.reps):
                prefix = toks(P)
                cold[P].append(ttft(prefix + toks(SUFFIX)))
                stored[P].append(prefix)
                llm.reset_prefix_cache()
        time.sleep(10)                                     # let asynchronous writes reach the files
        fadvise_dir(Path(a.disk_dir))
        flushed, t0 = 0, time.perf_counter()
        for f in sorted(Path(a.flush_dir).rglob("*")):
            if flushed >= a.flush_gb * 2**30 or not f.is_file() or f.stat().st_size < 2**20:
                continue
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                while flushed < a.flush_gb * 2**30:
                    b = os.read(fd, 64 << 20)
                    if not b:
                        break
                    flushed += len(b)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        flush_gbps = flushed / (time.perf_counter() - t0) / 1e9
        fadvise_dir(Path(a.disk_dir))
        for P in LENGTHS:
            hit = []
            for prefix in stored[P]:
                hit.append(ttft(prefix + toks(SUFFIX)))
                llm.reset_prefix_cache()
            res[str(P)] = {"cold_with_store": {"median_ms": round(statistics.median(cold[P]), 3),
                                               "all_ms": [round(x, 3) for x in cold[P]]},
                           "disk_cold": {"median_ms": round(statistics.median(hit), 3),
                                         "all_ms": [round(x, 3) for x in hit]}}
            print("disk-cold", P, {k: v["median_ms"] for k, v in res[str(P)].items()}, flush=True)
        res["flush"] = {"bytes": flushed, "gbps": round(flush_gbps, 3)}
    for P in (() if (a.mode == "disk" and a.disk_cold) else LENGTHS):
        cold, hit = [], []
        for _ in range(a.reps):
            prefix = toks(P)
            cold.append(ttft(prefix + toks(SUFFIX)))      # prefill; with LMCache also stores
            if a.mode == "none":
                hit.append(ttft(prefix + toks(SUFFIX)))   # GPU prefix-cache hit
            else:
                llm.reset_prefix_cache()                  # drop the GPU copy; LMCache keeps its own
                if a.mode == "disk":
                    fadvise_dir(Path(a.disk_dir))
                hit.append(ttft(prefix + toks(SUFFIX)))   # load from LMCache's tier
            llm.reset_prefix_cache()
        key_cold = "recompute" if a.mode == "none" else "cold_with_store"
        key_hit = {"none": "gpu", "cpu": "cpu", "disk": "disk"}[a.mode]
        res[str(P)] = {key_cold: {"median_ms": round(statistics.median(cold), 3), "all_ms": [round(x, 3) for x in cold]},
                       key_hit: {"median_ms": round(statistics.median(hit), 3), "all_ms": [round(x, 3) for x in hit]}}
        print(a.mode, P, {k: v["median_ms"] for k, v in res[str(P)].items()}, flush=True)

    def _cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable ({type(e).__name__})"
    import torch
    import vllm
    from importlib.metadata import PackageNotFoundError, version

    def _ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "not installed"
    # A Hugging Face cache path is .../models--<org>--<name>/snapshots/<revision>: keep both parts, so a
    # rerun can pin the same weights (review 2026-11-09, R5: only the revision's basename was kept).
    parts = Path(a.model).parts
    repo = next((q[len("models--"):].replace("--", "/") for q in parts if q.startswith("models--")), a.model)
    rev = parts[-1] if "snapshots" in parts else "unknown"
    out = {"mode": a.mode, "model": Path(a.model).name if "/" in a.model else a.model,
           "model_repo": repo, "model_revision": rev,
           "lengths": list(LENGTHS), "suffix_tokens": SUFFIX, "reps": a.reps, "results": res,
           "env": {"vllm": vllm.__version__, "torch": torch.__version__, "python": platform.python_version(),
                   "transformers": _ver("transformers"), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0),
                   "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=index,name,pcie.link.gen.current,"
                                       "pcie.link.width.current,utilization.gpu,memory.used",
                                       "--format=csv,noheader"])}}
    if a.mode != "none":
        out["env"]["lmcache"] = _ver("lmcache")      # lmcache has no __version__; the old probe wrote "unknown"
        secret = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        out["lmcache_env"] = {k: ("<redacted>" if any(x in k for x in secret) else v)
                              for k, v in os.environ.items() if k.startswith("LMCACHE_")}
    if a.mode == "disk":
        out["disk_mount"] = _cmd(["findmnt", "-no", "SOURCE,FSTYPE", "-T", a.disk_dir])
        out["disk_cpu_buffer_gb"] = a.disk_cpu_buffer_gb
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
