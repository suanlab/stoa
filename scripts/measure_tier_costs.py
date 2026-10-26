#!/usr/bin/env python3
"""Measure what moving a KV block between tiers costs on a real machine (notebook §B, §P).

The cost model prices a tier read as overhead + bytes / bandwidth (`stoa.calibration`), with device
profiles that are declared assumptions; the online replay (§AT H6, §AV) shows that whether any policy
that moves blocks pays is set by the price of a migration, which nothing here had measured. This script
measures, for KV-block-sized transfers on the machine it runs on:

  path         what it is                                        maps to
  gpu_copy     device-to-device copy (HBM read + write)          bw_gpu, oh_gpu
  h2d / d2h    pinned host <-> GPU (cudaMemcpyAsync)             bw_cpu, oh_cpu; GPU<->CPU migration
  disk_read    file -> pinned host -> GPU, page cache evicted    bw_disk, oh_disk; promotion from disk
  disk_write   GPU -> pinned host -> file, fsync                 demotion to disk
  overlap      a GEMM stream alone vs. with a concurrent copy    the price of an *asynchronous* move
               stream (H2D + D2H), both slowdown and copy rate

Each path is timed at several block sizes and fitted as latency = overhead + bytes / bandwidth, so the
result drops into `calibration.DeviceProfile` directly. Nothing here reads the traces, and nothing is
written outside --dir and --out.

Run on the GPU server (only PyTorch with CUDA and numpy are needed):
    python3 measure_tier_costs.py --dir /path/on/the/disk/tier --out tier_costs.json
--dir must be on the device that would hold the disk tier (local NVMe, not tmpfs); it needs about
4 x the largest block size free (default sizes: < 1 GB). --no-gpu measures only the host <-> disk paths.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# KV bytes per token = 2 (K, V) x layers x kv_heads x head_dim x 2 bytes (bf16); same shapes as
# stoa.calibration.MODELS, repeated here so the script runs without the repository.
MODELS = {"llama3-8b-like": (32, 8, 128), "llama3-70b-like": (80, 8, 128)}
BLOCK_TOKENS = (16, 64, 512)          # Bailian blocks, kv-cache-tester blocks, Mooncake blocks


def kv_bytes(model: str, tokens: int) -> int:
    layers, heads, dim = MODELS[model]
    return 2 * layers * heads * dim * 2 * tokens


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001 -- metadata is best effort, never fatal
        return f"unavailable ({type(e).__name__})"


def machine_info(dirpath: Path, torch) -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": _cmd(["sh", "-c", "lscpu | grep 'Model name' | head -1"]),
            "dir": str(dirpath),
            "dir_mount": _cmd(["findmnt", "-no", "SOURCE,FSTYPE", "-T", str(dirpath)]),
            "block_devices": _cmd(["lsblk", "-d", "-o", "NAME,ROTA,SIZE,MODEL"])}
    if torch is not None:
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,"
                                       "pcie.link.gen.current,pcie.link.width.current,utilization.gpu,memory.used",
                                       "--format=csv,noheader"])
    return info


def fit(points: list[tuple[int, float]]) -> dict:
    """Least squares latency_ms = overhead_ms + bytes / (bw_GBps * 1e9) * 1e3."""
    xs = [b / 1e9 for b, _ in points]            # GB
    ys = [ms / 1e3 for _, ms in points]          # s
    n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    icept = my - slope * mx
    return {"bandwidth_GBps": round(1 / slope, 3) if slope > 0 else None,
            "overhead_ms": round(icept * 1e3, 4), "n_sizes": n}


def timed(fn, repeats: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 4), "p10_ms": round(ts[len(ts) // 10], 4),
            "p90_ms": round(ts[min(len(ts) - 1, (9 * len(ts)) // 10)], 4), "repeats": repeats}


def evict(path: Path) -> None:
    """Drop the file's clean pages from the page cache (no root needed), so a read hits the device."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="directory on the disk tier (not tmpfs)")
    ap.add_argument("--out", default="tier_costs.json")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--no-gpu", action="store_true", help="host <-> disk paths only")
    ap.add_argument("--overlap-seconds", type=float, default=5.0)
    a = ap.parse_args()

    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None
    gpu = (not a.no_gpu) and torch is not None and torch.cuda.is_available()
    if not a.no_gpu and not gpu:
        print("No CUDA device visible; rerun with --no-gpu to measure host <-> disk only.")
        return 2

    d = Path(a.dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    sizes = sorted({kv_bytes(m, t) for m in MODELS for t in BLOCK_TOKENS})
    need = 2 * max(sizes) + (64 << 20)
    if shutil.disk_usage(d).free < need:
        print(f"{d} has {shutil.disk_usage(d).free >> 20} MiB free; need {need >> 20} MiB.")
        return 2

    out = {"machine": machine_info(d, torch), "sizes_bytes": sizes,
           "models": {m: {"kv_bytes_per_token": kv_bytes(m, 1)} for m in MODELS},
           "block_tokens": list(BLOCK_TOKENS), "paths": {}, "fits": {}}
    dev = torch.device("cuda") if gpu else None
    sync = torch.cuda.synchronize if gpu else (lambda: None)

    def host(n):
        if torch is not None:
            return torch.empty(n, dtype=torch.uint8, pin_memory=gpu)
        return np.empty(n, dtype=np.uint8)

    def as_np(t):
        return t.numpy() if torch is not None else t

    for n in sizes:
        rows = {}
        h = host(n)
        as_np(h)[:] = np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8)
        f = d / f"stoa_block_{n}.bin"
        if gpu:
            g = torch.empty(n, dtype=torch.uint8, device=dev)
            g2 = torch.empty(n, dtype=torch.uint8, device=dev)

            def _gcopy():
                g2.copy_(g)
                sync()

            def _h2d():
                g.copy_(h, non_blocking=True)
                sync()

            def _d2h():
                h.copy_(g, non_blocking=True)
                sync()
            rows["gpu_copy"] = timed(_gcopy, a.repeats)
            rows["h2d"] = timed(_h2d, a.repeats)
            rows["d2h"] = timed(_d2h, a.repeats)

        def _write():
            if gpu:
                h.copy_(g, non_blocking=True)
                sync()
            with open(f, "wb", buffering=0) as fh:
                fh.write(memoryview(as_np(h)))
                fh.flush()
                os.fsync(fh.fileno())
        rows["disk_write"] = timed(_write, a.repeats)

        def _read():
            evict(f)
            with open(f, "rb", buffering=0) as fh:
                fh.readinto(memoryview(as_np(h)))
            if gpu:
                g.copy_(h, non_blocking=True)
                sync()
        rows["disk_read"] = timed(_read, a.repeats)
        f.unlink()
        out["paths"][str(n)] = rows
        print(f"{n / 2**20:9.2f} MiB  " + "  ".join(f"{k} {v['median_ms']:.3f} ms" for k, v in rows.items()),
              flush=True)

    for path in next(iter(out["paths"].values())):
        out["fits"][path] = fit([(int(n), r[path]["median_ms"]) for n, r in out["paths"].items()])

    if gpu:
        # The price of an asynchronous move: a GEMM loop on one stream, alone and with a copy stream
        # moving the largest block back and forth. Reports the compute slowdown and the copy rate.
        # bf16 on Ampere or newer; fp16 elsewhere (pre-Ampere cuBLAS has no fast bf16 GEMM)
        gdt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        m = torch.randn(8192, 8192, device=dev, dtype=gdt)
        comp, copy = torch.cuda.Stream(), torch.cuda.Stream()
        n = max(sizes)
        h, g = host(n), torch.empty(n, dtype=torch.uint8, device=dev)

        def gemm_rate(with_copy: bool) -> tuple[float, float]:
            sync()
            iters, moved, t0 = 0, 0, time.perf_counter()
            while time.perf_counter() - t0 < a.overlap_seconds:
                with torch.cuda.stream(comp):
                    for _ in range(4):
                        m @ m
                if with_copy:
                    with torch.cuda.stream(copy):
                        g.copy_(h, non_blocking=True)
                        h.copy_(g, non_blocking=True)
                    moved += 2 * n
                comp.synchronize()
                iters += 4
            copy.synchronize()
            dt = time.perf_counter() - t0
            return iters / dt, moved / dt / 1e9
        alone, _ = gemm_rate(False)
        both, gbps = gemm_rate(True)
        out["overlap"] = {"gemm_iters_per_s_alone": round(alone, 2),
                          "gemm_iters_per_s_with_copy": round(both, 2),
                          "compute_slowdown_pct": round(100 * (1 - both / alone), 2),
                          "copy_GBps_under_compute": round(gbps, 3),
                          "block_bytes": n, "gemm": f"8192x8192 {str(gdt).split('.')[-1]}, 4 per round"}
        print("overlap:", out["overlap"])

    # The price of an asynchronous move on the serving path: a migration overlaps compute (above), but
    # it shares a link with demand reads. A demand thread loads blocks back to back; a background
    # thread migrates blocks back to back. Price per migration, in demand-read time:
    #   (demand loads/s alone - demand loads/s with migrations) / migrations/s x demand latency.
    # Same-direction traffic (a promotion while blocks are loaded) and opposite-direction traffic (a
    # demotion; PCIe is full duplex) are measured separately, as is cpu->disk under disk reads.
    import threading

    def contention(demand, migrate, seconds: float) -> dict:
        def run(fn, stop, box):
            k = 0
            while not stop.is_set():
                fn()
                k += 1
            box.append(k)

        def rate(fn_pair):
            stop, boxes, ths = threading.Event(), [], []
            for fn in fn_pair:
                box = []
                boxes.append(box)
                ths.append(threading.Thread(target=run, args=(fn, stop, box)))
            t0 = time.perf_counter()
            for t in ths:
                t.start()
            time.sleep(seconds)
            stop.set()
            for t in ths:
                t.join()
            dt = time.perf_counter() - t0
            return [b[0] / dt for b in boxes]
        d0, = rate([demand])
        d1, m1 = rate([demand, migrate])
        lat = 1e3 / d0
        price = (d0 - d1) / m1 * lat if m1 > 0 else float("nan")
        return {"demand_per_s_alone": round(d0, 2), "demand_per_s_with_migration": round(d1, 2),
                "migrations_per_s": round(m1, 2), "demand_latency_ms": round(lat, 4),
                "price_per_migration_ms": round(price, 4),
                "demand_slowdown_pct": round(100 * (1 - d1 / d0), 2)}

    nb = kv_bytes("llama3-8b-like", 512)             # a Mooncake block of the 8B shape
    out["contention"] = {"block_bytes": nb, "seconds": a.overlap_seconds}
    if gpu:
        hd, gd = host(nb), torch.empty(nb, dtype=torch.uint8, device=dev)     # demand buffers
        hm, gm = host(nb), torch.empty(nb, dtype=torch.uint8, device=dev)     # migration buffers
        sd, sm = torch.cuda.Stream(), torch.cuda.Stream()

        def on(stream, fn):
            def f():
                with torch.cuda.stream(stream):
                    fn()
                stream.synchronize()
            return f
        load = on(sd, lambda: gd.copy_(hd, non_blocking=True))
        promote = on(sm, lambda: gm.copy_(hm, non_blocking=True))      # cpu -> gpu, same direction
        demote = on(sm, lambda: hm.copy_(gm, non_blocking=True))       # gpu -> cpu, opposite
        out["contention"]["h2d_load_vs_promotion"] = contention(load, promote, a.overlap_seconds)
        out["contention"]["h2d_load_vs_demotion"] = contention(load, demote, a.overlap_seconds)
        for k in ("h2d_load_vs_promotion", "h2d_load_vs_demotion"):
            print(k, out["contention"][k], flush=True)

        # The thread-based measurement above can serialize the two streams itself (each thread
        # launches, then blocks in synchronize), which would make full-duplex links look shared. This
        # one enqueues K demand loads and K migrations on two streams from ONE thread, with no
        # synchronization in between, and times each stream with CUDA events: if the copy engines
        # overlap, the demand stream finishes no later than it does alone.
        K = 40

        def enqueue(mig):
            torch.cuda.synchronize()
            e0d, e1d = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0m, e1m = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0d.record(sd)
            e0m.record(sm)
            for _ in range(K):
                with torch.cuda.stream(sd):
                    gd.copy_(hd, non_blocking=True)
                if mig is not None:
                    with torch.cuda.stream(sm):
                        mig()
            e1d.record(sd)
            e1m.record(sm)
            torch.cuda.synchronize()
            return e0d.elapsed_time(e1d), e0m.elapsed_time(e1m)

        enqueue(None)                                                    # warm up
        ev = {}
        for name, mig in (("promotion", lambda: gm.copy_(hm, non_blocking=True)),
                          ("demotion", lambda: hm.copy_(gm, non_blocking=True))):
            alone = statistics.median(enqueue(None)[0] for _ in range(5))
            pairs = [enqueue(mig) for _ in range(5)]
            withm = statistics.median(p[0] for p in pairs)
            ev[f"h2d_load_vs_{name}"] = {
                "k": K, "demand_ms_alone": round(alone, 3), "demand_ms_with_migration": round(withm, 3),
                "migration_stream_ms": round(statistics.median(p[1] for p in pairs), 3),
                "price_per_migration_ms": round((withm - alone) / K, 4),
                "demand_latency_ms": round(alone / K, 4)}
            print("events", name, ev[f"h2d_load_vs_{name}"], flush=True)
        out["contention_events"] = ev
        out["copy_engines"] = _cmd(["sh", "-c", "nvidia-smi -q -i 0 | grep -iE 'copy|async' | head -5"])
    fd_, fm_ = d / "stoa_demand.bin", d / "stoa_migrate.bin"
    hdd, hmm = host(nb), host(nb)
    for f_, hb in ((fd_, hdd), (fm_, hmm)):
        with open(f_, "wb", buffering=0) as fh:
            fh.write(memoryview(as_np(hb)))
            os.fsync(fh.fileno())

    def disk_load():
        evict(fd_)
        with open(fd_, "rb", buffering=0) as fh:
            fh.readinto(memoryview(as_np(hdd)))

    def disk_demote():
        with open(fm_, "wb", buffering=0) as fh:
            fh.write(memoryview(as_np(hmm)))
            os.fsync(fh.fileno())
    out["contention"]["disk_load_vs_demotion"] = contention(disk_load, disk_demote,
                                                            max(a.overlap_seconds, 10.0))
    print("disk_load_vs_demotion", out["contention"]["disk_load_vs_demotion"], flush=True)
    fd_.unlink()
    fm_.unlink()
    if gpu:
        out["other_gpus_at_end"] = _cmd(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                         "--format=csv,noheader"])

    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
