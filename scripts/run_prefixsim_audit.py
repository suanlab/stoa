#!/usr/bin/env python3
"""Does a published evaluator's headline survive the matching rule serving engines implement? (§BK)

PrefixSim (cacheMon/PrefixSim, the simulator of "When Fancy Eviction Fails", arXiv 2609.28870) counts a
cached block as a hit even when an earlier block of the same request missed ("A block after a miss still
counts", doc/design.md). Its own documentation notes that vLLM and SGLang stop matching at a request's first
miss. Its README headline -- RandomCompute saves 10.4 points more prefill compute than LRU at 16k blocks on
the freeinference trace -- is scored under the first rule. scripts/patches/prefixsim_first_miss.patch adds a
second score from the same phase-1 match: hits up to the request's first miss only. The cache's behaviour
is unchanged; only the count differs.

Registered before running (research notes, 2026-11-16): the headline does not survive engine-style scoring if
RandomCompute's lead over LRU at 16k blocks on freeinference falls below 1 point or reverses; it survives if it
stays at 5 points or more.

Steps (each writes only under DIR, outside the repository, except `collect`):
  setup DIR   clone PrefixSim at PREFIXSIM_COMMIT (libCacheSim at LIBCACHESIM_COMMIT), apply the patch, build;
              download the two traces at DATASET_REVISION and check their SHA-256. The build needs CMake,
              Ninja, pkg-config, glib, zstd and xxhash; without root they come from conda-forge, e.g.
              `micromamba create -p ENV -c conda-forge cmake ninja pkg-config glib zstd xxhash gperftools
              gxx_linux-64=13 sysroot_linux-64` and then PATH, PKG_CONFIG_PATH, CC and CXX pointed at ENV (run
              this script with a Python that has huggingface_hub, not the toolchain environment's).
  run DIR     the sweep (ALGOS x SIZES x both traces, `--jobs` in parallel; caches smaller than a trace's
              largest request load the whole trace, about 13 GiB each), the hole dumps of HOLE_ALGOS at
              HOLE_CACHE on freeinference, and freeinference converted to JSONL for request lengths.
              About 3 hours on 8 jobs of a 2.7 GHz Xeon. `--quick` runs one size and two policies.
  mooncake DIR  our two Mooncake traces through the same simulator (notebook BL): converted with traceConvLLM,
              then libCacheSim's own implementations of the plain and adaptive policies of section 5.9 at our four
              tier sizes (2/5/10/20% of distinct blocks), hit counting (`--cost-model uniform`), both scores.
              Writes experiments/prefixsim_mooncake.json. Minutes; needs `setup` first and data/ populated.
  collect DIR experiments/prefixsim_matching.json from DIR's logs. It also checks the patch against PrefixSim's
              own hole dump: the hits after each request's first hole, summed from the dump, must equal the
              difference of the two hit ratios for every dumped policy.
Usage: python3 scripts/run_prefixsim_audit.py {setup,run,collect,mooncake} DIR [--jobs 8] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIXSIM_REPO = "https://github.com/cacheMon/PrefixSim"
PREFIXSIM_COMMIT = "160d5f81724dd9ec28b10d6abb64615e6adbb332"
LIBCACHESIM_COMMIT = "3fa480b6894ed1b9f75b8a9c737a2e34c2ca6016"
DATASET = "harvardMadsys/2026_prefixcache"
DATASET_REVISION = "ba89a343a048073dda72617bb6087910da6214d0"
TRACE_SHA256 = {"freeinference": "39f75bbba1ce912546517802719ef7742d0db7fa53dcbc66681bba090aa5d1f7",
                "chutes": "da57f89c74db817c121d24d3d6e4bc415eb7dd54aa72548c6971a991bd99970b"}
PATCH = ROOT / "scripts" / "patches" / "prefixsim_first_miss.patch"
COST_MODEL = "qwen3coder30b_blksz_16"
ALGOS = ("lru", "s3fifo", "sieve", "arc", "random_compute", "gdsf_compute", "partial_node_random_compute",
         "workload_aware", "belady", "belady_compute")
SIZES = ("16k", "32k", "64k", "128k", "256k")
HOLE_ALGOS = ("lru", "random_compute", "partial_node_random_compute")
HOLE_CACHE = 16384
OUT = ROOT / "experiments" / "prefixsim_matching.json"
OUT_MOONCAKE = ROOT / "experiments" / "prefixsim_mooncake.json"
MOONCAKE = {"conversation": "data/mooncake_conversation_trace.jsonl", "toolagent": "data/mooncake_toolagent_trace.jsonl"}
MOONCAKE_ALGOS = ("lru", "lfu", "arc", "lecar", "cacheus", "s3fifo", "sieve", "wtinylfu", "belady")
MOONCAKE_FRACS = (0.02, 0.05, 0.10, 0.20)       # run_reactive_real.FRACS: max(1, int(frac * distinct blocks))


def sh(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def setup(d: Path):
    src = d / "PrefixSim"
    if not src.exists():
        sh(["git", "clone", "-q", PREFIXSIM_REPO, src])
    sh(["git", "checkout", "-q", PREFIXSIM_COMMIT], cwd=src)
    sh(["git", "submodule", "update", "--init", "--recursive", "-q"], cwd=src)
    got = subprocess.run(["git", "-C", "libCacheSim", "rev-parse", "HEAD"], cwd=src, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert got == LIBCACHESIM_COMMIT, f"libCacheSim at {got}, expected {LIBCACHESIM_COMMIT}"
    if subprocess.run(["git", "apply", "--check", PATCH], cwd=src).returncode == 0:
        sh(["git", "apply", PATCH], cwd=src)
    else:
        assert subprocess.run(["git", "apply", "--reverse", "--check", PATCH], cwd=src).returncode == 0, \
            "the patch neither applies nor is already applied"
    sh(["cmake", "-G", "Ninja", "-B", "_build", "-DCMAKE_BUILD_TYPE=Release"], cwd=src)
    sh(["ninja", "-C", "_build", "prefixsim", "traceConvLLM"], cwd=src)
    from huggingface_hub import snapshot_download
    snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION, local_dir=str(src / "traces"),
                      allow_patterns=["*.lcp.zst"])
    for t, want in TRACE_SHA256.items():
        got = sha256(src / "traces" / f"{t}.lcp.zst")
        assert got == want, f"{t}: SHA-256 {got} != pinned {want}"
    print("setup done; traces verified", flush=True)


def run(d: Path, jobs: int, quick: bool):
    src = d / "PrefixSim"
    bin_ = src / "_build" / "bin" / "prefixsim"
    algos, sizes = (("lru", "random_compute"), ("16k",)) if quick else (ALGOS, SIZES)

    def one(job):
        t, s, a, extra = job
        r = subprocess.run([str(bin_), "--trace", str(src / "traces" / f"{t}.lcp.zst"), "--cache-size", s,
                            "--algo", a, "--cost-model", COST_MODEL, *extra], capture_output=True, text=True)
        return [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    sweep = [(t, s, a, []) for t in TRACE_SHA256 for s in sizes for a in algos]
    (d / "holes").mkdir(exist_ok=True)
    holes = [("freeinference", f"{HOLE_CACHE // 1024}k", a, ["--dump-holes", str(d / "holes") + "/"])
             for a in (HOLE_ALGOS if not quick else algos)]
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(one, sweep + holes))
    (d / "results.log").write_text("\n".join(l for r in res[:len(sweep)] for l in r) + "\n")
    (d / "holes" / "results.log").write_text("\n".join(l for r in res[len(sweep):] for l in r) + "\n")
    sh([src / "_build" / "bin" / "traceConvLLM", "--input", src / "traces" / "freeinference.lcp.zst",
        "--output", d / "freeinference.jsonl"])


def parse(log: Path):
    rows = []
    for line in log.read_text().splitlines():
        if not line.startswith("RESULT"):
            continue
        kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
        rows.append({"trace": Path(kv["trace"]).name.split(".")[0], "cache_blocks": int(kv["cache_size"]),
                     "algo": kv["algo"], "requests": int(kv["n_req"]), "skipped": int(kv["n_req_skipped"]),
                     "block_hit_ratio": float(kv["block_hit_ratio"]),
                     "prefix_block_hit_ratio": float(kv["prefix_block_hit_ratio"]),
                     "compute_saving_ratio": float(kv["compute_saving_ratio"]),
                     "prefix_compute_saving_ratio": float(kv["prefix_compute_saving_ratio"]),
                     "unexpected_evictions": int(kv["n_unexpected_eviction"])})
    return rows


def collect(d: Path):
    rows = parse(d / "results.log")
    assert all(r["unexpected_evictions"] == 0 for r in rows), "a run PrefixSim itself marks untrustworthy"
    assert all(r["prefix_block_hit_ratio"] <= r["block_hit_ratio"] + 1e-12 for r in rows)
    hole_rows = {r["algo"]: r for r in parse(d / "holes" / "results.log")}
    sweep = {(r["trace"], r["cache_blocks"], r["algo"]): r for r in rows}
    for a, r in hole_rows.items():                       # the dump runs reproduce the sweep's numbers
        s = sweep[("freeinference", HOLE_CACHE, a)]
        assert all(abs(s[k] - r[k]) < 1e-9 for k in ("block_hit_ratio", "prefix_block_hit_ratio",
                                                      "compute_saving_ratio", "prefix_compute_saving_ratio"))
    lengths = []
    with open(d / "freeinference.jsonl") as fh:
        for line in fh:
            hs = json.loads(line)["hash_ids"]
            if len(set(hs)) <= HOLE_CACHE:               # requests larger than the cache are skipped
                lengths.append(len(hs))
    pat = re.compile(r"\(idx: (\d+), len: (\d+)\)")
    holes = {}
    for a in hole_rows:
        f = next((d / "holes").glob(f"holes_*_cache{HOLE_CACHE}_{a}.txt"))
        lost = total = n = nh = mid = 0
        for i, line in enumerate(open(f)):
            hs = [(int(x), int(y)) for x, y in pat.findall(line)]
            total += lengths[i]
            n += 1
            nh += len(hs)
            if hs:
                l = (lengths[i] - hs[0][0]) - sum(y for _, y in hs)
                lost += l
                mid += l > 0
        assert n == hole_rows[a]["requests"] - hole_rows[a]["skipped"] or n == hole_rows[a]["requests"], \
            f"{a}: {n} hole lines for {hole_rows[a]['requests']} requests"
        patch_lost = hole_rows[a]["block_hit_ratio"] - hole_rows[a]["prefix_block_hit_ratio"]
        assert abs(lost / total - patch_lost) < 1e-5, f"{a}: hole dump {lost / total} != patch {patch_lost}"
        holes[a] = {"holes_per_request": round(nh / n, 4), "share_requests_hit_after_hole": round(mid / n, 4),
                    "hits_after_first_hole_share_of_blocks": round(lost / total, 6)}
    OUT.write_text(json.dumps({
        "config": {"prefixsim_commit": PREFIXSIM_COMMIT, "libcachesim_commit": LIBCACHESIM_COMMIT,
                   "dataset": DATASET, "dataset_revision": DATASET_REVISION, "trace_sha256": TRACE_SHA256,
                   "patch": "scripts/patches/prefixsim_first_miss.patch", "cost_model": COST_MODEL,
                   "scores": {"gap": "PrefixSim's own: every resident block of a request is a hit",
                              "first_miss": "hits up to the request's first miss only (vLLM / SGLang matching)"},
                   "registered": "headline does not survive if random_compute - lru at 16k on freeinference "
                                 "under first-miss scoring is < 1 point; survives if >= 5",
                   "hole_check": f"freeinference, {HOLE_CACHE} blocks; patch agrees with the dump to 1e-5"},
        "rows": rows, "holes": holes}, indent=2) + "\n")
    print("wrote", OUT.relative_to(ROOT))


def mooncake(d: Path, jobs: int):
    src = d / "PrefixSim"
    conv, sim = src / "_build" / "bin" / "traceConvLLM", src / "_build" / "bin" / "prefixsim"
    jobs_ = []
    distinct = {}
    for t, path in MOONCAKE.items():
        lcp = d / f"mooncake_{t}.lcp"
        r = subprocess.run([str(conv), "--input", str(ROOT / path), "--output", str(lcp), "--block-size-tokens", "512"],
                           capture_output=True, text=True, check=True)
        distinct[t] = int(re.search(r"(\d+) unique", r.stdout + r.stderr).group(1))
        for f in MOONCAKE_FRACS:
            c = max(1, int(f * distinct[t]))
            jobs_ += [(t, f, c, a, f"{lcp}.zst") for a in MOONCAKE_ALGOS]

    def one(j):
        t, f, c, a, lcp = j
        out = subprocess.run([str(sim), "--trace", lcp, "--cache-size", str(c), "--algo", a, "--cost-model", "uniform"],
                             capture_output=True, text=True, check=True).stdout
        kv = dict(x.split("=", 1) for x in next(l for l in out.splitlines() if l.startswith("RESULT")).split()[1:]
                  if "=" in x)
        return {"trace": t, "cache_frac": f, "cache_blocks": c, "algo": a,
                "block_hit_ratio": float(kv["block_hit_ratio"]),
                "prefix_block_hit_ratio": float(kv["prefix_block_hit_ratio"]),
                "unexpected_evictions": int(kv["n_unexpected_eviction"])}
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(one, jobs_))
    assert all(r["unexpected_evictions"] == 0 for r in rows)
    OUT_MOONCAKE.write_text(json.dumps({
        "config": {"prefixsim_commit": PREFIXSIM_COMMIT, "libcachesim_commit": LIBCACHESIM_COMMIT,
                   "patch": "scripts/patches/prefixsim_first_miss.patch", "cost_model": "uniform (hit counting)",
                   "traces": MOONCAKE, "distinct_blocks": distinct, "fracs": list(MOONCAKE_FRACS),
                   "note": "libCacheSim's LFU resets a block's count when it is evicted; ours keeps it"},
        "rows": rows}, indent=2) + "\n")
    print("wrote", OUT_MOONCAKE.relative_to(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("setup", "run", "collect", "mooncake"))
    ap.add_argument("dir")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    {"setup": lambda: setup(d), "run": lambda: run(d, a.jobs, a.quick), "collect": lambda: collect(d),
     "mooncake": lambda: mooncake(d, a.jobs)}[a.step]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
