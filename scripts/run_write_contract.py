#!/usr/bin/env python3
"""Store admission replayed inside a serving stack, under two write-path contracts (notebook §BM).

The additive model prices a store as a per-request constant. Whether that constant is what a request pays
depends on the connector's write path:

  blocking   LMCache 0.3.9 in-process (`LMCacheConnectorV1`): `wait_for_save` blocks the step until the
             KV is saved to host memory
  async      vLLM 0.11's `OffloadingConnector`: the store is submitted with `transfer_async` and the step
             goes on

`serving_replay/matrix.sh` replays a slice of the Bailian traceA trace open-loop against a fresh vLLM
server per run, with one of three per-request store policies (store-all; oracle: skip the store of a
request that creates no block read again later in the slice; store-none), and records each request's
time to first token. `--collect DIR` reads those runs (DIR/stage1_cpu40, DIR/stage2_cpu120,
DIR/stage2_cpu40, and the two workload files, for their digests) into experiments/write_contract_a6000.json:
per-request TTFTs of the measured requests, LMCache's logged store costs, and per-request hit tokens from
both connectors' logs. DIR/tiers holds measure_serving_tiers.py's runs on the same GPU (none.json,
cpu.json, and cpu.log for LMCache's logged costs), the host's per-block prices as §BE derived them for
the A100. DIR/sglang holds SGLang 0.5.10.post1 HiCache runs (cpu0: no HiCache; cpu40, cpu120: host tier
size) of its three write policies, and DIR/rep the replicates at 120 GB and 0.35x (SGLang pairs, one more run
of each vLLM cell, and SGLang pairs with cudaMemcpy transfers). SGLang reports one cached-token count per
request (GPU and host together), read from the response's usage. Without --collect it re-derives the summary
from that file; no GPU needed.

Hit tokens per request are the GPU-local prefix hit and the host-tier hit at the request's last lookup:
LMCache logs both ("LMCache hit tokens", "need to load"); the offloading connector logs them through
`serving_replay/skipstore_connector.py` (STOA_HITS). Stores carry no request id, so LMCache's store time
is counted from the first measured request's lookup on (a few warm-up requests may still be in flight).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "write_contract_a6000.json"
STAGES = {"stage1_cpu40": (1, 40, "traceA1200", 400), "stage2_cpu120": (2, 120, "traceA600", 200),
          "stage2_cpu40": (2, 40, "traceA600", 200)}
TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\] LMCache INFO:\S* "
STORE = re.compile(TS + r"Stored (\d+) out of total \d+ tokens.*?cost ([\d.]+) ms")
LOOKUP = re.compile(TS + r"Reqid: (\S+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (-?\d+)")
OFFHIT = re.compile(r"STOA_HITS reqs=\d+ measured_reqs=(\d+) measured_tokens=(\d+) measured_gpu_hit=(\d+) "
                    r"measured_host_hit=(\d+)")
SKIP = re.compile(r"STOA_SKIPSTORE calls=\d+ store_jobs=\d+ stored_gpu_blocks=(\d+) skipped_reqs=\d+")
HARNESS = ("client.py", "skipstore_connector.py", "make_workload.py", "client_sgl.py")
SGL_POLICIES = ("write_through", "write_through_selective", "write_back")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_run(stem: Path, warm: int) -> dict:
    conn, policy, speed = re.match(r"(\w+)_([\w-]+)_x([\d.]+)$", stem.name).groups()
    recs = [json.loads(line) for line in open(f"{stem}.jsonl")]
    log = Path(f"{stem}.server.log").read_text(errors="ignore")
    run = {"connector": conn, "policy": policy, "speed": float(speed),
           "ttft_ms": [r["ttft_ms"] for r in recs if not r["warmup"]],   # None: the request failed
           "errors": sum(1 for r in recs if r["err"]), "hits": None}
    if conn == "lmcache":
        looks = LOOKUP.findall(log)
        first, last = {}, {}
        for ts, rid, tot, hit, need in looks:
            first.setdefault(rid, ts)
            last[rid] = (int(tot), int(hit), int(need))
        order = sorted(first, key=first.get)
        t0 = first[order[warm]]
        meas = [last[r] for r in order[warm:]]
        stores = [(int(n), float(c)) for ts, n, c in STORE.findall(log) if ts >= t0]
        run["lmcache_store"] = {"ms_measured": round(sum(c for _, c in stores), 3), "stores": len(stores),
                                "tokens": sum(n for n, _ in stores)}
        run["hits"] = {"reqs": len(meas), "tokens": sum(m[0] for m in meas),
                       "gpu_hit": sum(m[1] - m[2] for m in meas), "host_hit": sum(max(m[2], 0) for m in meas)}
    elif conn == "offload":
        n, tok, g, h = map(int, OFFHIT.findall(log)[-1])
        run["hits"] = {"reqs": n, "tokens": tok, "gpu_hit": g, "host_hit": h}
        run["offload_stored_gpu_blocks"] = int(SKIP.findall(log)[-1])
    return run


def parse_sgl(stem: Path) -> dict:
    recs = [json.loads(line) for line in open(f"{stem}.jsonl")]
    meas = [r for r in recs if not r["warmup"]]
    ok = [r for r in meas if r["ttft_ms"] is not None]
    return {"ttft_ms": [r["ttft_ms"] for r in meas], "errors": sum(1 for r in recs if r["err"]),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
            "cached_tokens": sum(r["cached_tokens"] or 0 for r in ok)}


def collect(d: Path) -> None:
    runs = {}
    for stage, (_s, _cap, _wl, warm) in STAGES.items():
        for f in sorted((d / stage).glob("*.jsonl")):
            stem = f.with_suffix("")
            runs[f"{stage}/{stem.name}"] = parse_run(stem, warm)
    workloads = {}
    for name in ("traceA1200", "traceA600"):
        rows = [json.loads(line) for line in open(d / f"{name}.jsonl")]
        meas = [r for r in rows if not r["warmup"]]
        src = ROOT / "data" / "bailian_qwen_traceA_blksz_16.jsonl"
        workloads[name] = {"source": src.name, "source_sha256": sha256(src),
                           "sha256": sha256(d / f"{name}.jsonl"), "requests": len(rows), "warmup": len(rows) - len(meas),
                           "span_s": round(rows[-1]["t"] - rows[0]["t"], 1),
                           "oracle_store_share_measured": round(sum(r["store_oracle"] for r in meas) / len(meas), 4)}
    for f in sorted((d / "sglang").glob("cpu*/sgl_*.jsonl")) + sorted((d / "rep").glob("*/sgl_*.jsonl")):
        stem = f.with_suffix("")
        runs[str(stem.relative_to(d))] = parse_sgl(stem)
    for f in sorted((d / "rep").glob("vllm_*/*.jsonl")):
        stem = f.with_suffix("")
        runs[str(stem.relative_to(d))] = parse_run(stem, 200)
    tiers = {m: json.loads((d / "tiers" / f"{m}.json").read_text()) for m in ("none", "cpu")}
    for t in tiers.values():
        t.pop("lmcache_env", None)
    tlog = (d / "tiers" / "cpu.log").read_text(errors="ignore")
    logged = {k: [float(m.group(2)) for m in pat.finditer(tlog) if m.group(1) == "8192"]
              for k, pat in (("cpu_retrieve_ms_8192", re.compile(r"Retrieved (\d+) out of \d+ required tokens.*?cost "
                                                                 r"([\d.]+) ms")),
                             ("cpu_store_ms_8192", re.compile(r"Stored (\d+) out of total \d+ tokens.*?cost ([\d.]+) ms")))}
    here = Path(__file__).resolve().parent / "serving_replay"
    OUT.write_text(json.dumps({
        "producer": "scripts/serving_replay/matrix.sh on one RTX A6000 48 GB (PCIe Gen3 x16, driver 570.133.07; "
                    "GPU 0 of the same host ran another vLLM server at full utilisation throughout); "
                    "assembled by run_write_contract.py --collect",
        "stack_sglang": {"sglang": "0.5.10.post1", "torch": "2.9.1+cu128", "flashinfer": "0.6.7.post3",
                         "max_total_tokens": 73000, "page_size": 1, "attention_backend": "flashinfer",
                         "hicache_io_backend": "kernel (rep/io_direct: direct)", "chunked_prefill_size": 4096},
        "stack": {"vllm": "0.11.0", "lmcache": "0.3.9", "torch": "2.8.0",
                  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B@6a6f4aa4197940add57724a7707d069478df56b1",
                  "gpu_memory_utilization": 0.53, "lmcache_chunk_tokens": 256, "max_model_len": 24576},
        # byte-identical to what ran; run_one.sh as run (e434a43f...) differed from the committed one only in
        # defaulting STOA_ENV / STOA_HF to the GPU host's paths
        "harness_sha256": {n: sha256(here / n) for n in HARNESS},
        "stages": {k: {"stage": s, "host_tier_gb": cap, "workload": wl, "warmup": w}
                   for k, (s, cap, wl, w) in STAGES.items()},
        "workloads": workloads, "runs": runs,
        "tier_probe": {"runs": tiers, "lmcache_logged": logged}}, separators=(",", ":")) + "\n")
    print("wrote", OUT)


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(d: dict) -> dict:
    rows = {}
    for key, r in d["runs"].items():
        stage = key.split("/")[0]
        if stage not in STAGES:
            continue
        t = [x for x in r["ttft_ms"] if x is not None]
        h = r["hits"]
        rows[(stage, r["connector"], r["policy"], r["speed"])] = {
            "p50_ms": pct(t, 0.5), "p90_ms": pct(t, 0.9), "p99_ms": pct(t, 0.99), "measured": len(t),
            "errors": r["errors"],
            "host_hit_share": round(h["host_hit"] / h["tokens"], 4) if h else None,
            "gpu_hit_share": round(h["gpu_hit"] / h["tokens"], 4) if h else None,
            "store_ms_per_request": round(r["lmcache_store"]["ms_measured"] / len(r["ttft_ms"]), 2)
            if "lmcache_store" in r else None,
            "offload_stored_gpu_blocks": r.get("offload_stored_gpu_blocks")}
    g = lambda *k: rows[k]                                                          # noqa: E731

    def gain(stage, conn, speed, base="store-all", alt="oracle"):
        a, o = g(stage, conn, base, speed)["p50_ms"], g(stage, conn, alt, speed)["p50_ms"]
        return {"ms": round(a - o, 1), "pct": round(100 * (a - o) / a, 1)}

    out = {"rows": [{"stage": k[0], "connector": k[1], "policy": k[2], "speed": k[3], **v}
                    for k, v in sorted(rows.items())]}
    # the contract alone: storing everything vs storing nothing, at (almost) equal hits
    # (pct of store-none's p50: how much storing everything adds)
    out["store_all_minus_store_none"] = {
        f"{c}@{s}": {"ms": round(g("stage1_cpu40", c, "store-all", s)["p50_ms"]
                                 - g("stage1_cpu40", c, "store-none", s)["p50_ms"], 1),
                     "pct": round(100 * (g("stage1_cpu40", c, "store-all", s)["p50_ms"]
                                         / g("stage1_cpu40", c, "store-none", s)["p50_ms"] - 1), 1)}
        for c in ("lmcache", "offload") for s in (0.35, 0.2)}
    out["oracle_gain"] = {f"{st}/{c}@{s}": gain(st, c, s) for st in STAGES for c in ("lmcache", "offload")
                          for s in ((0.35, 0.2, 0.1) if st == "stage1_cpu40" else (0.35, 0.2))}
    saved = {f"{st}@{s}": round(g(st, "lmcache", "store-all", s)["store_ms_per_request"]
                                - g(st, "lmcache", "oracle", s)["store_ms_per_request"], 2)
             for st in STAGES for s in ((0.35, 0.2, 0.1) if st == "stage1_cpu40" else (0.35, 0.2))}
    out["lmcache_store_ms_saved_per_request"] = saved
    og = out["oracle_gain"]
    # registered predictions (notebook §BM; research notes 2026-11-16 and 2026-11-21, before the runs)
    out["checks"] = {
        "b_blocking_gain_exceeds_store_saved": all(og[f"{k.split('@')[0]}/lmcache@{k.split('@')[1]}"]["ms"] >= v
                                                   for k, v in saved.items()),
        "b_blocking_gain_grows_with_load": all(
            og[f"{st}/lmcache@0.35"]["ms"] > og[f"{st}/lmcache@0.2"]["ms"] for st in STAGES),
        "c_async_within_5pct_at_120gb": all(abs(og[f"stage2_cpu120/offload@{s}"]["pct"]) < 5 for s in (0.35, 0.2)),
        "c_prime_async_oracle_faster_at_40gb": all(og[f"{st}/offload@{s}"]["ms"] > 0 for st in
                                                   ("stage1_cpu40", "stage2_cpu40") for s in (0.35, 0.2)),
        "blocking_gain_exceeds_async_gain_at_40gb": all(
            og[f"{st}/lmcache@{s}"]["ms"] > og[f"{st}/offload@{s}"]["ms"] for st in ("stage1_cpu40", "stage2_cpu40")
            for s in (0.35, 0.2)),
        # within 0.1 point of the measured tokens (offload at 0.2x: 0.1215 vs 0.1220)
        "equal_host_hits_at_120gb": all(abs(g("stage2_cpu120", c, "store-all", s)["host_hit_share"]
                                            - g("stage2_cpu120", c, "oracle", s)["host_hit_share"]) <= 0.001
                                        for c in ("lmcache", "offload") for s in (0.35, 0.2))}
    # do the hit part (async gain at 40 GB) and the store part (blocking gain at 120 GB) add up to the
    # blocking gain at 40 GB? (at 0.35x within 5%; at 0.2x the blocking gain is larger than the sum)
    out["decomposition_stage2"] = {
        f"@{s}": {"hit_part_ms": og[f"stage2_cpu40/offload@{s}"]["ms"],
                  "store_part_ms": og[f"stage2_cpu120/lmcache@{s}"]["ms"],
                  "blocking_gain_40gb_ms": og[f"stage2_cpu40/lmcache@{s}"]["ms"]} for s in (0.35, 0.2)}
    # per 512-token block from the 8,192-token prefix, as run_serving_tiers.py derives the A100's
    tp = d["tier_probe"]
    r8, r5 = tp["runs"]["none"]["results"]["8192"], tp["runs"]["none"]["results"]["512"]
    med = lambda xs: sorted(xs)[len(xs) // 2]                                       # noqa: E731
    pb = {"recompute_ms": round((r8["recompute"]["median_ms"] - r8["gpu"]["median_ms"]) / 16, 3),
          "gpu_marginal_ms": round((r8["gpu"]["median_ms"] - r5["gpu"]["median_ms"]) / 15, 3),
          "cpu_load_ms": round(med(tp["lmcache_logged"]["cpu_retrieve_ms_8192"]) / 16, 3),
          "cpu_store_ms": round(med(tp["lmcache_logged"]["cpu_store_ms_8192"]) / 16, 3)}
    # the reuse probability above which storing a new block pays: S / (R - (g + L))
    pb["store_break_even_reuse"] = round(pb["cpu_store_ms"] / (pb["recompute_ms"] - pb["gpu_marginal_ms"]
                                                               - pb["cpu_load_ms"]), 4)
    out["per_512_token_block"] = pb

    # SGLang HiCache: p50 and cached share per (host GB, write policy, speed); the host part estimated against
    # no HiCache at the same load (same GPU pool)
    sg = {}
    for key, r in d["runs"].items():
        if not key.startswith("sglang/"):
            continue
        cap = int(key.split("/")[1][3:])
        pol, sp = key.split("/")[2][4:].rsplit("_x", 1)
        sg[(cap, pol, float(sp))] = {"p50_ms": pct([x for x in r["ttft_ms"] if x is not None], 0.5),
                                     "cached_share": r["cached_tokens"] / r["prompt_tokens"], "errors": r["errors"]}
    for (cap, pol, sp), v in sg.items():
        v["host_share_est"] = None if pol == "none" else v["cached_share"] - sg[(0, "none", sp)]["cached_share"]
    out["sglang_rows"] = [{"host_gb": k[0], "policy": k[1], "speed": k[2], **v} for k, v in sorted(sg.items())]
    rel = lambda a, b: 100 * (a - b) / b                                             # noqa: E731
    p = lambda c, q, s: sg[(c, q, s)]["p50_ms"]                                      # noqa: E731
    h = lambda c, q, s: sg[(c, q, s)]["host_share_est"]                              # noqa: E731
    # registered before the runs from a simulation of the documented semantics (a write on the second access);
    # recorded as outcomes, three of them failures
    out["sglang_registered"] = {
        "S1_selective_host_hits_below_a_fifth": {"held": all(h(120, "write_through_selective", s) * 5
                                                             <= min(h(120, q, s) for q in ("write_through", "write_back"))
                                                             for s in (0.35, 0.2))},
        "S2_selective_slower_by_5pct_at_120gb": {
            "pct": {str(s): round(rel(p(120, "write_through_selective", s), p(120, "write_through", s)), 1)
                    for s in (0.35, 0.2)}},
        "S3_write_back_penalty_grows_with_load": {
            "ms": {str(s): round(p(120, "write_back", s) - p(120, "write_through", s), 1) for s in (0.35, 0.2)}},
        "S4_host_hits_below_4pct_at_40gb": {"held": all(h(40, q, 0.35) < 0.04 for q in SGL_POLICIES)}}
    r2 = out["sglang_registered"]
    r2["S2_selective_slower_by_5pct_at_120gb"]["held"] = all(v >= 5 for v in r2["S2_selective_slower_by_5pct_at_120gb"]["pct"].values())
    m3 = r2["S3_write_back_penalty_grows_with_load"]["ms"]
    r2["S3_write_back_penalty_grows_with_load"]["held"] = m3["0.35"] > m3["0.2"] > 0

    # replicates at 120 GB, 0.35x
    def p50(key):
        return pct([x for x in d["runs"][key]["ttft_ms"] if x is not None], 0.5)
    pairs = {"kernel": [[p50("sglang/cpu120/sgl_write_through_x0.35"), p50("sglang/cpu120/sgl_write_through_selective_x0.35")]]
             + [[p50(f"rep/sgl_r{i}/sgl_write_through_x0.35"), p50(f"rep/sgl_r{i}/sgl_write_through_selective_x0.35")]
                for i in (1, 2)],
             "direct": [[p50(f"rep/io_direct/sgl_write_through_x0.35_direct_r{i}"),
                         p50(f"rep/io_direct/sgl_write_through_selective_x0.35_direct_r{i}")] for i in (1, 2)]}
    out["sglang_write_timing"] = {k: {"pairs_ms": v, "selective_faster_pct": [round(100 * (1 - b / a), 1) for a, b in v]}
                                  for k, v in pairs.items()}
    vr = {f"{c}/{q}": (g("stage2_cpu120", c, q, 0.35)["p50_ms"], p50(f"rep/vllm_r1/{c}_{q}_x0.35"))
          for c in ("lmcache", "offload") for q in ("store-all", "oracle")}
    sgr = {q: [p50(f"sglang/cpu120/sgl_{q}_x0.35")] + [p50(f"rep/sgl_r{i}/sgl_{q}_x0.35") for i in (1, 2)]
           for q in ("write_through", "write_through_selective")}
    moves = [abs(b / a - 1) for a, b in vr.values()] + [abs(x / v[0] - 1) for v in sgr.values() for x in v[1:]]
    out["replicates"] = {
        "vllm_120gb_0.35": {k: {"first_ms": a, "replicate_ms": b} for k, v in vr.items() for a, b in [v]},
        "rerun_moves_pct": [round(100 * min(moves), 1), round(100 * max(moves), 1)],
        "async_oracle_gain_pct": [round(100 * (1 - vr["offload/oracle"][i] / vr["offload/store-all"][i]), 1) for i in (0, 1)],
        "blocking_oracle_gain_pct": [round(100 * (1 - vr["lmcache/oracle"][i] / vr["lmcache/store-all"][i]), 1)
                                     for i in (0, 1)]}
    return out


def main() -> int:
    if "--collect" in sys.argv:
        collect(Path(sys.argv[sys.argv.index("--collect") + 1]))
    d = json.loads(OUT.read_text())
    d["summary"] = summarize(d)
    OUT.write_text(json.dumps(d, separators=(",", ":")) + "\n")
    s = d["summary"]
    for r in s["rows"]:
        print(f"{r['stage']:<14} {r['connector']:<8} {r['policy']:<10} {r['speed']:>5} p50 {r['p50_ms']:>8.1f} "
              f"host hit {r['host_hit_share']}")
    print(json.dumps({k: s[k] for k in ("store_all_minus_store_none", "lmcache_store_ms_saved_per_request",
                                        "checks", "decomposition_stage2", "per_512_token_block",
                                        "sglang_registered", "sglang_write_timing", "replicates")}, indent=1))
    for r in s["sglang_rows"]:
        print(f"sglang {r['host_gb']:>4} {r['policy']:<24} {r['speed']:>5} p50 {r['p50_ms']:>8.1f} "
              f"cached {r['cached_share']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
