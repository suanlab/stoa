#!/usr/bin/env python3
"""External replication (H7): the split-instant ceiling and the arrival arms on Alibaba's Qwen traces.

Every number in the paper was measured on Mooncake (one provider, Kimi). Alibaba released
anonymized KV-block traces from a Qwen serving cluster on Bailian (Wang et al., ATC'25,
https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon, Apache-2.0). This script runs the
UNMODIFIED producers on them:

    run_reachable_ceiling.py   the split-instant ceiling and unreachable share
    run_arrival_admission.py   the five arrival arms under both tie conventions

each under the full action space and under tier-only (`run_tier_only.py`'s restriction, imported
rather than copied so the definition cannot drift).

How the producers are made to read Bailian without editing them:

  * Schema. A Bailian record is a SUPERSET of a Mooncake record (chat_id, parent_chat_id, type and
    turn are extra), so conversion is a projection to {timestamp, input_length, output_length,
    hash_ids}, validated: timestamps non-decreasing, hash ids kept verbatim and in order, and the
    16-token identity len(hash_ids) == ceil(input_length / 16) checked per record. Timestamps keep
    Bailian's unit (seconds; Mooncake's are ms). No producer reads the timestamp.
  * Block size. `stoa.mooncake.BLOCK_TOKENS` is set to 16 in the worker process before any trace
    is loaded, so the loader's own identity check validates the converted files at 16 tokens and
    every block is sized 16 * BYTES_PER_TOKEN bytes. A wrapper around the producer's
    `load_mooncake` asserts that size on every item. NOTE: block size enters the cost model ONLY
    through `size_bytes`, and only through tier capacity (a fraction of total bytes). All blocks
    are the same size, so the capacity in BLOCKS is the same at 16 or 512 tokens; the per-access
    cost tables (`simulator._TIER_READ_MS`, `_REPR_TOKENS`) charge per block access and are not
    scaled by tokens per block. The results are therefore invariant to the constant; setting it
    to 16 is about letting the loader check the data, not about changing the costs.
  * Trace table. Both producers read a module-level `TRACES` dict inside `main()`. Each worker
    imports the producer as a module (so `main()` does not run), replaces `TRACES` with the one
    Bailian trace it is assigned, sets `sys.argv`, and calls `main()`. `runpy` cannot do this:
    the module body reassigns `TRACES` after any `init_globals` are applied.
  * Session. `run_arrival_admission.py` infers a session from a request's first block
    (`train_session_features.parse_requests`). Bailian carries real chat ids, but they are NOT
    used: the replication keeps the producer's features exactly as they are.

Data: the four files are Git-LFS objects; they are fetched from media.githubusercontent.com and
checked against the LFS sha256. They are stored as data/bailian_*.jsonl, which `.gitignore`
(data/*.jsonl) already excludes -- `data/bailian/` is NOT ignored, so it is not used.

Usage (CPU; full traces, 112-core host, ~1-2 h wall with --jobs 16):
    python3 scripts/run_bailian.py                  # download, convert, run everything, merge
    python3 scripts/run_bailian.py --hours 1        # first hour of each trace only
    python3 scripts/run_bailian.py --birthx         # §AS correction: birth access = compulsory miss
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"

BLOCK_TOKENS = 16
SOURCE = "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon"
MEDIA = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/"
# name -> (upstream file, Git-LFS sha256 from the pointer committed upstream)
BAILIAN = {
    "traceA": ("qwen_traceA_blksz_16.jsonl",
               "07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17"),
    "traceB": ("qwen_traceB_blksz_16.jsonl",
               "68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59"),
    "coder": ("qwen_coder_blksz_16.jsonl",
              "3d74974cebb7bccb69b1ca6ea210a44d9a3033443c0271ed015327284dea1aff"),
    "thinking": ("qwen_thinking_blksz_16.jsonl",
                 "41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906"),
}
MOONCAKE = {"conversation": "data/mooncake_conversation_trace.jsonl",
            "toolagent": "data/mooncake_toolagent_trace.jsonl",
            "synthetic": "data/mooncake_synthetic_trace.jsonl"}
SPACES = ("full", "tier_only")
SPLITS = (0.3, 0.4, 0.5, 0.6, 0.7)
MOONCAKE_FIELDS = ("timestamp", "input_length", "output_length", "hash_ids")


def raw_path(name: str) -> Path:
    return DATA / f"bailian_{BAILIAN[name][0]}"


def converted_path(name: str, hours: float, group: int = 1) -> Path:
    tag = (f"_first{hours:g}h" if hours else "") + (f"_g{group}" if group > 1 else "")
    return DATA / f"bailian_{name}{tag}_mooncake.jsonl"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> Path:
    fname, sha = BAILIAN[name]
    dst = raw_path(name)
    if not dst.exists():
        print(f"downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(MEDIA + fname, dst)
    got = _sha256(dst)
    if got != sha:
        raise SystemExit(f"{dst}: sha256 {got} != LFS oid {sha}")
    return dst


def convert(src, dst, hours: float = 0.0, block_tokens: int = BLOCK_TOKENS,
            group: int = 1) -> dict:
    """Project Bailian records onto the Mooncake schema, in file order, validating as it goes.

    `group` > 1 is a SENSITIVITY variant, not the replication: each run of `group` consecutive
    16-token ids becomes one coarser block, identified -- like Mooncake's -- by its content AND
    everything before it in the request (a prefix chain), so `group=32` gives 512-token blocks
    that satisfy Mooncake's own identity ceil(input_length / 512). Ids are then new integers;
    only the default `group=1` keeps the upstream ids verbatim.

    `hours` > 0 keeps the requests with timestamp < hours * 3600 (seconds since trace start): a
    time PREFIX, never a sample of requests, which would break the reuse structure being measured.
    Raises on a timestamp that goes backwards (the prefix would not be a prefix) and on a record
    whose block count violates the `block_tokens` identity.
    """
    src, dst = Path(src), Path(dst)
    n, prev, t0 = 0, None, None
    chain: dict = {}
    with src.open() as fi, dst.open("w") as fo:
        for line_no, line in enumerate(fi):
            rec = json.loads(line)
            t = rec["timestamp"]
            if prev is not None and t < prev:
                raise ValueError(f"{src}:{line_no + 1}: timestamp {t} < previous {prev}")
            prev = t
            t0 = t if t0 is None else t0
            if hours and t - t0 >= hours * 3600.0:
                break
            hs = rec["hash_ids"]
            if hs and len(hs) != math.ceil(rec["input_length"] / block_tokens):
                raise ValueError(f"{src}:{line_no + 1}: {len(hs)} blocks for "
                                 f"{rec['input_length']} tokens at {block_tokens} tokens/block")
            out = {k: rec[k] for k in MOONCAKE_FIELDS}
            if group > 1:
                key, ids = None, []
                for k in range(0, len(hs), group):
                    key = chain.setdefault((key, tuple(hs[k:k + group])), len(chain))
                    ids.append(key)
                assert len(ids) == math.ceil(rec["input_length"] / (block_tokens * group))
                out["hash_ids"] = ids
            fo.write(json.dumps(out) + "\n")
            n += 1
    return {"requests": n, "first_timestamp": t0, "last_timestamp": prev if n else None}


def structure(path) -> dict:
    """Block-level shape of a trace in the Mooncake schema (no block-size assumption)."""
    cnt: Counter = Counter()
    n, ts = 0, []
    with Path(path).open() as fh:
        for line in fh:
            rec = json.loads(line)
            cnt.update(rec["hash_ids"])
            ts.append(rec["timestamp"])
            n += 1
    blocks, acc = len(cnt), sum(cnt.values())
    return {"requests": n, "blocks": blocks, "accesses": acc,
            "singleton_block_frac": round(sum(1 for v in cnt.values() if v == 1) / blocks, 4),
            "reused_access_frac": round(1 - blocks / acc, 4),
            "mean_accesses_per_block": round(acc / blocks, 3),
            "max_accesses_per_block": max(cnt.values()),
            "mean_blocks_per_request": round(acc / n, 1),
            "timestamp_span": round(ts[-1] - ts[0], 3)}


# --------------------------------------------------------------------------------- worker ---

def _load_producer(script: str):
    spec = importlib.util.spec_from_file_location(f"_h7_{Path(script).stem}", SCRIPTS / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_birth_excluded() -> None:
    """The birth-access correction (notebook §AS), identical to scripts/run_birth_excluded.py.

    `Workload.future_counts(split)` counts a block's creation access when the block is first seen
    at or after the split, and the placement code bills it as a tier read; it is a compulsory miss
    (prefill) that no tier can serve. Replicated here rather than imported because that module
    patches `Action.space` at import time depending on `sys.argv`. Applied at class level before
    any producer is imported.
    """
    from stoa.traces import Workload
    if getattr(Workload.future_counts, "_birth_excluded", False):
        return
    orig = Workload.future_counts

    def future_counts_birth_excluded(self, split_step):
        f = orig(self, split_step)
        return {i: (n - 1 if self.traces[i].access_steps[0] >= split_step and n > 0 else n)
                for i, n in f.items()}

    future_counts_birth_excluded._birth_excluded = True
    Workload.future_counts = future_counts_birth_excluded


def patch_birth_billed() -> None:
    """The pre-§AS convention, made EXPLICIT: every access at or after the split is billed,
    creation included. `src/stoa/traces.py` changed its default to birth-excluded on 2026-10-19
    (23:12 KST), while this replication was running; without this patch the birth-billed mode
    would silently follow whatever the source default is. All birth-billed parts in
    experiments/bailian_*.json were computed by workers started before that change.
    """
    from stoa.traces import Workload

    def future_counts_birth_billed(self, split_step):
        return {i: sum(1 for s in tr.access_steps if s >= split_step)
                for i, tr in self.traces.items()}

    future_counts_birth_billed._birth_excluded = False
    Workload.future_counts = future_counts_birth_billed


def post_split_never_reused(path, splits) -> dict:
    """Per split: of the blocks first seen at or after `time_split(frac)`, the fraction accessed
    exactly once (only their birth). Request index is the timestep, as in `load_mooncake`."""
    first, count, last = {}, Counter(), -1
    with Path(path).open() as fh:
        for t, line in enumerate(fh):
            hs = json.loads(line)["hash_ids"]
            if hs:
                last = t
            for h in hs:
                first.setdefault(h, t)
                count[h] += 1
    lo = min(first.values())
    out = {}
    for sf in splits:
        st = max(1, int(lo + sf * (last - lo)))
        new = [h for h, f in first.items() if f >= st]
        out[str(sf)] = {"new_blocks": len(new),
                        "never_reused_frac": round(sum(1 for h in new if count[h] == 1) / len(new), 4)
                        if new else None}
    return out


def worker(trace: str, space: str, producer: str, data_path: str, out: str,
           splits: list[float], block_tokens: int = BLOCK_TOKENS, birthx: bool = False) -> None:
    """One (trace, action space, producer) run in a fresh process, so patches cannot leak."""
    for p in (str(ROOT / "src"), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    patch_birth_billed()                 # explicit either way, never the source default
    if birthx:
        patch_birth_excluded()
    if space == "tier_only":
        import run_tier_only  # noqa: F401  -- its import patches Action.space; main() is not run
        from stoa.environment import Action, Representation, Timing
        sp = Action.space()
        assert sp and all(a.representation == Representation.PLAINTEXT
                          and a.timing == Timing.ONLINE for a in sp), "tier-only patch missing"
    import stoa.mooncake as mc
    mc.BLOCK_TOKENS = block_tokens
    import stoa.gnn
    from stoa.environment import Action
    assert len(stoa.gnn.LEGAL_ACTIONS) == len(Action.space()), "LEGAL_ACTIONS built before patch"

    mod = _load_producer(producer)
    orig = mod.load_mooncake

    def checked_load(path, *a, **k):
        wl = orig(path, *a, **k)
        want = block_tokens * mc.BYTES_PER_TOKEN
        assert all(it.size_bytes == want for it in wl.items), "a block is mis-sized"
        return wl

    mod.load_mooncake = checked_load
    costs: dict = {}
    if hasattr(mod, "prefix_greedy_cost"):          # the arrival producer's reference; recorded
        pgc = mod.prefix_greedy_cost                # so the frame check below can compare it

        def recorded_pgc(*a, **k):
            costs["prefix_greedy_cost"] = v = pgc(*a, **k)
            return v

        mod.prefix_greedy_cost = recorded_pgc
    mod.TRACES = {trace: data_path}
    argv = [producer, "--out", out]
    if producer == "run_reachable_ceiling.py":
        argv += ["--splits", *[str(s) for s in splits]]
    sys.argv = argv
    t0 = time.time()
    rc = mod.main()
    if rc not in (0, None):
        raise SystemExit(rc)
    if costs:
        Path(out + ".costs.json").write_text(json.dumps(costs) + "\n")
    print(f"[{trace}/{space}/{producer} {splits if 'reach' in producer else ''}] "
          f"done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------- merge ---

def _h7(reach: dict, arrival: dict) -> dict:
    """Per trace: unreachable > 85% at split 0.5 in BOTH action spaces, and every order arm under
    all four (space x tie) configurations captures >= 3x that space's split-time ceiling -- the
    form in which the paper states the Mooncake result (notebook §AR.3)."""
    out = {}
    for name in BAILIAN:
        rec = {"configs": {}}
        ok_unreach, ok_ratio = True, True
        for space in SPACES:
            row = next(r for r in reach[space]["traces"][name]["rows"] if r["split_frac"] == 0.5)
            ceil_pct, unreach = row["reachable_pct"], row["unreachable_pct"]
            ok_unreach &= unreach is not None and unreach > 85.0
            for tie in ("new-behind", "new-ahead"):
                rows = [r for r in arrival[space]["results"] if r["trace"] == name
                        and r.get("tie") == tie]
                order = [r["captured_pct"] for r in rows if r["axis"] == "order"]
                mag = {r["arm"]: r["captured_pct"] for r in rows if r["axis"] == "magnitude"}
                ratio = (min(order) / ceil_pct) if ceil_pct and ceil_pct > 0 else None
                ok = ceil_pct is not None and min(order) >= 3.0 * ceil_pct
                ok_ratio &= ok
                rec["configs"][f"{space}/{tie}"] = {
                    "split_ceiling_pct": ceil_pct, "unreachable_pct": unreach,
                    "order_arms_min": min(order), "order_arms_max": max(order),
                    "constant_mean": mag.get("constant-mean"),
                    "constant_median": mag.get("constant-median"),
                    "min_order_over_ceiling": round(ratio, 2) if ratio is not None else None,
                    "order_ge_3x_ceiling": ok}
        rec["unreachable_gt_85_both_spaces"] = ok_unreach
        rec["order_ge_3x_ceiling_all_configs"] = ok_ratio
        rec["h7_holds_on_trace"] = ok_unreach and ok_ratio
        out[name] = rec
    n = sum(r["h7_holds_on_trace"] for r in out.values())
    return {"criterion": ("per trace: split-0.5 unreachable share > 85% under both action spaces AND "
                          "min over the three order arms >= 3x the split-time ceiling (reachable_pct "
                          "at 0.5) under all four space x tie configurations; H7 holds if >= 3 of 4 "
                          "traces pass"),
            "traces_passing": n, "h7": "holds" if n >= 3 else "falsified", "per_trace": out}


def _frame_check(reach: dict, parts: Path, traces) -> dict:
    """The two producers bill their reference from different routines: `place_with_beliefs` on
    prefix counts (reachable ceiling) and `prefix_greedy_cost` (arrival). The 3x comparison divides
    one producer's percentage by the other's, so their references must coincide. Recorded here
    rather than assumed (the ratios are over different gaps if they do not)."""
    out = {}
    for s in SPACES:
        for n in traces:
            row = next(r for r in reach[s]["traces"][n]["rows"] if r["split_frac"] == 0.5)
            c = json.loads((parts / f"arrival_{n}_{s}.json.costs.json").read_text())
            pg = c["prefix_greedy_cost"]
            out[f"{n}/{s}"] = {"reach_reference_cost": row["reference_cost"],
                               "arrival_prefix_greedy_cost": round(pg, 1),
                               "agree_to_0.1": abs(row["reference_cost"] - pg) <= 0.1 + 1e-9}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=0.0,
                    help="keep only the first H hours of each trace (0 = full two hours)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--traces", nargs="+", default=list(BAILIAN))
    ap.add_argument("--worker", nargs=5, metavar=("TRACE", "SPACE", "PRODUCER", "DATA", "OUT"))
    ap.add_argument("--splits", type=float, nargs="+", default=list(SPLITS))
    ap.add_argument("--group", type=int, default=1,
                    help="sensitivity only: merge N consecutive 16-token ids into one chained block")
    ap.add_argument("--block-tokens", type=int, default=BLOCK_TOKENS, help=argparse.SUPPRESS)
    ap.add_argument("--birthx", action="store_true",
                    help="bill a post-split block's birth access as a compulsory miss (§AS); "
                         "writes *_birthx.json")
    ap.add_argument("--parts", type=str, default="",
                    help="directory for per-run outputs and logs (default: $TMPDIR/stoa_bailian_parts)")
    ap.add_argument("--out-prefix", type=str, default="experiments/bailian")
    args = ap.parse_args()

    if args.worker:
        trace, space, producer, data_path, out = args.worker
        worker(trace, space, producer, data_path, out, args.splits, args.block_tokens, args.birthx)
        return 0

    conv, shape = {}, {}
    for name in args.traces:
        raw = fetch(name)
        dst = converted_path(name, args.hours, args.group)
        conv[name] = convert(raw, dst, args.hours, group=args.group)
        conv[name]["file"] = str(dst.relative_to(ROOT))
        shape[name] = structure(dst)
        shape[name]["post_split_new_blocks"] = post_split_never_reused(dst, args.splits)
        print(f"{name}: {shape[name]}", flush=True)
    mooncake_shape = {n: {**structure(ROOT / p),
                          "post_split_new_blocks": post_split_never_reused(ROOT / p, args.splits)}
                      for n, p in MOONCAKE.items() if (ROOT / p).exists()}

    # Per-run outputs and logs live OUTSIDE the tree (experiments/ is tracked, data/<dir>/ is not
    # ignored). Re-running with the same --parts resumes: finished parts are not recomputed.
    parts = Path(args.parts) if args.parts else Path(tempfile.gettempdir()) / "stoa_bailian_parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in args.traces:
        data_path = str(ROOT / conv[name]["file"])
        for space in SPACES:
            for sf in args.splits:
                jobs.append((name, space, "run_reachable_ceiling.py", data_path,
                             str(parts / f"reach_{name}_{space}_{sf}.json"), [sf]))
            jobs.append((name, space, "run_arrival_admission.py", data_path,
                         str(parts / f"arrival_{name}_{space}.json"), []))
    # Longest first: arrival on the biggest traces dominates the wall clock.
    jobs.sort(key=lambda j: (j[2] != "run_arrival_admission.py", -shape[j[0]]["blocks"]))

    def run(job):
        name, space, producer, data_path, out, splits = job
        if Path(out).exists():
            return 0
        log = Path(out).with_suffix(".log")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", name, space, producer,
               data_path, out, "--block-tokens", str(BLOCK_TOKENS * args.group)]
        if args.birthx:
            cmd.append("--birthx")
        if splits:
            cmd += ["--splits", *map(str, splits)]
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"{'ok ' if rc == 0 else 'FAIL'} {name} {space} {producer} {splits} -> {log.name}",
              flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rcs = list(ex.map(run, jobs))
    if any(rcs):
        raise SystemExit(f"{sum(1 for r in rcs if r)} run(s) failed; see {parts}/*.log")

    reach = {s: {"traces": {}} for s in SPACES}
    arrival = {s: {"results": []} for s in SPACES}
    for name in args.traces:
        for space in SPACES:
            rows = []
            for sf in args.splits:
                d = json.loads((parts / f"reach_{name}_{space}_{sf}.json").read_text())
                rows += d["traces"][name]["rows"]
            u = [r["unreachable_pct"] for r in rows if r["unreachable_pct"] is not None]
            reach[space]["traces"][name] = {"rows": rows, "unreachable_min": min(u) if u else None,
                                            "unreachable_max": max(u) if u else None}
            d = json.loads((parts / f"arrival_{name}_{space}.json").read_text())
            arrival[space]["results"] += d["results"]
            arrival[space]["producer_config"] = d["config"]

    config = {
        "source": SOURCE, "license": "Apache-2.0 (upstream LICENSE and README); data not redistributed",
        "citation": "Wang et al., KVCache Cache in the Wild, USENIX ATC'25",
        "files": {n: {"upstream": BAILIAN[n][0], "lfs_sha256": BAILIAN[n][1]} for n in args.traces},
        "block_tokens": BLOCK_TOKENS * args.group,
        "regrouped": (f"{args.group} consecutive 16-token ids per block, prefix-chained "
                      "(sensitivity, not the replication)" if args.group > 1 else None),
        "block_size_in_cost_model": ("size_bytes = 16 * BYTES_PER_TOKEN for every block; size enters "
                                     "only tier capacity (a fraction of total bytes), so capacity in "
                                     "blocks is invariant to the constant; per-access costs are per "
                                     "block, not per token"),
        "subsample": (f"first {args.hours:g} h by timestamp" if args.hours else "none (full traces)"),
        "conversion": conv,
        "structure": shape, "mooncake_structure_same_function": mooncake_shape,
        "action_spaces": {"full": "Action.space() unmodified",
                          "tier_only": "run_tier_only.py's restriction (PLAINTEXT/ONLINE), imported"},
        "producers": ["scripts/run_reachable_ceiling.py", "scripts/run_arrival_admission.py"],
        "session_proxy": "first block of a request (producer's parse_requests); chat_id unused",
        "birth_access": ("compulsory miss: excluded from future counts for blocks first seen at or "
                         "after the split (§AS; patch identical to run_birth_excluded.py)"
                         if args.birthx else "billed as a tier read (pre-§AS convention, "
                         "enforced by patch_birth_billed regardless of the src default)"),
    }
    sfx = "_birthx" if args.birthx else ""
    pre = ROOT / args.out_prefix
    (pre.parent / f"{pre.name}_reachable{sfx}.json").write_text(json.dumps(
        {"config": {**config, "splits": args.splits, "reference": "prefix_greedy"},
         **reach}, indent=2) + "\n")
    h7 = _h7(reach, arrival) if 0.5 in args.splits and set(args.traces) == set(BAILIAN) else None
    (pre.parent / f"{pre.name}_arrival{sfx}.json").write_text(json.dumps(
        {"config": config, **arrival,
         "frame_check": _frame_check(reach, parts, args.traces) if 0.5 in args.splits else None,
         "h7": h7}, indent=2) + "\n")
    print(f"wrote {pre.name}_reachable{sfx}.json and {pre.name}_arrival{sfx}.json")
    if h7:
        print(f"H7: {h7['h7']} ({h7['traces_passing']}/4 traces pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
