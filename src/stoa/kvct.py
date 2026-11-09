"""kv-cache-tester trace adapter: a SECOND, independent production-trace source.

Mooncake (mooncake.py) gives KV-block reuse from one serving stack. A characterization
that rests on a single source is not a characterization, so we add the anonymized
Claude Code agentic traces from kv-cache-tester (739 conversations,
github.com/callanjfox/kv-cache-tester), which differ from Mooncake in system, client,
and workload shape.

Schema (one JSON file per conversation):
    {"id", "block_size", "hash_id_scope": "local", "tool_tokens", "system_tokens",
     "requests": [{"t": <wall-clock seconds>, "in", "out", "hash_ids": [...]}, ...]}

Two properties matter for us:
  * `hash_id_scope` is **local** -- block ids repeat across conversations and mean
    different things, so ids MUST be namespaced by conversation before pooling. Failing
    to do so fabricates cross-conversation reuse (and inflates every hit rate).
  * `t` is a real wall-clock timestamp, so inter-arrival and think-time are available.
    Our Mooncake analysis concluded that reuse hinges on whether a session stays alive,
    a signal that trace could not express; this one can.

Pooling all conversations into one namespaced stream models a shared server cache,
which is the setting a tiering policy actually runs in.

Download (no GPU):
    python3 -c "from stoa.kvct import download; download(200)"

Acquisition is pinned and fails closed (review 2026-11-09, R5; notebook BJ). Files come from upstream
commit `UPSTREAM_COMMIT`, not the moving `master`; `data/kvct_SHA256SUMS` lists the 200 files the paper
uses with their SHA-256 (all 200 match the upstream blobs at that commit); the sampled universe is that
list, never whatever `trace_*.json` happens to sit in the directory; and a missing, truncated or altered
file raises instead of being skipped. Before this, a seed drew from a different population in a
directory with 739 files than in one with 200, and a corrupt file vanished silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .credit import AccessTrace
from .environment import ItemState, Representation, Tier
from .traces import Workload

UPSTREAM_COMMIT = "94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf"     # 2026-07-12
BASE_URL = f"https://raw.githubusercontent.com/callanjfox/kv-cache-tester/{UPSTREAM_COMMIT}/traces/"
DEFAULT_DIR = "data/kvct"
MANIFEST = Path(__file__).resolve().parents[2] / "data" / "kvct_SHA256SUMS"


def manifest(path: Path | None = None) -> dict[str, str]:
    """file name -> SHA-256 of every conversation the paper uses, in file order."""
    path = Path(path) if path is not None else MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: the kv-cache-tester universe is not defined without it")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            out[name] = digest
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
BYTES_PER_TOKEN = 2 * 2 * 32 * 128 * 2 // 1024      # illustrative; calibrate in M0-3


def download(n: int = 200, out_dir: str = DEFAULT_DIR, workers: int = 16) -> int:
    """Fetch the first `n` files of the manifest from the pinned commit and verify each one's SHA-256.
    Returns how many are present and verified; raises if any cannot be fetched or does not match."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    want = list(manifest().items())[:n]

    def one(item) -> str | None:
        name, digest = item
        path = Path(out_dir) / name
        if not (path.exists() and _sha256(path) == digest):
            urllib.request.urlretrieve(BASE_URL + name, path)
        return None if _sha256(path) == digest else name

    with ThreadPoolExecutor(workers) as ex:
        bad = [b for b in ex.map(one, want) if b]
    if bad:
        raise RuntimeError(f"{len(bad)} kv-cache-tester files do not match the manifest: {bad[:5]}")
    return len(want)


def _iter_conversations(path_dir: str, limit: int | None = None,
                        seed: int | None = None, offset: int = 0):
    """Iterate conversations, optionally as a RANDOM sample.

    `seed=None` keeps the historical sorted-prefix behaviour, which is what made two of
    our "independent samples" nested and produced a sign-reversed headline claim: file
    index correlates strongly with conversation length and reuse rate (mean requests per
    conversation falls 140 -> 37 from files 1-20 to 71-150). Pass a seed to draw a random
    subset, and vary the seed to obtain genuinely disjoint-in-distribution draws; `offset`
    additionally skips the first N files so successive draws can be made disjoint by
    construction.
    """
    want = manifest()
    files = [Path(path_dir) / name for name in sorted(want)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} kv-cache-tester files of the manifest are absent from "
                                f"{path_dir} (first: {missing[:3]}); run stoa.kvct.download()")
    files = files[offset:]
    if seed is not None:
        rng = random.Random(seed)
        files = rng.sample(files, min(limit or len(files), len(files)))
    elif limit is not None:
        files = files[:limit]
    for f in files:
        if _sha256(f) != want[f.name]:
            raise ValueError(f"{f} does not match its manifest SHA-256: truncated or altered")
        conv = json.loads(f.read_text())
        conv.setdefault("_file", f.name)
        yield conv


def load_kvct(path_dir: str = DEFAULT_DIR, limit: int | None = None,
              pool: bool = True, seed: int | None = None, offset: int = 0) -> Workload:
    """Load conversations as one `Workload`.

    Block ids are namespaced per conversation (`c{conv}:b{id}`) because the source
    scopes them locally; pooling without namespacing would invent shared blocks.
    Timesteps are the global order of requests sorted by wall-clock time, so reuse
    distance reflects real interleaving across concurrent conversations.
    """
    events: list[tuple[float, str, list[str]]] = []
    for conv in _iter_conversations(path_dir, limit, seed, offset):
        cid = conv.get("id", "c")
        prefix = f"{cid}:" if pool else ""
        for req in conv.get("requests", []):
            hs = req.get("hash_ids") or []
            if not hs:
                continue
            events.append((float(req.get("t", 0.0)), cid, [f"{prefix}b{h}" for h in hs]))

    events.sort(key=lambda e: e[0])                   # interleave conversations by time
    steps: dict[str, list[int]] = defaultdict(list)
    accesses: list[tuple[int, str]] = []
    for t, _cid, blocks in enumerate_events(events):
        for b in blocks:
            steps[b].append(t)
            accesses.append((t, b))

    horizon = len(events) or 1
    items = [
        ItemState(item_id=b, access_freq=len(s) / horizon,
                  size_bytes=64 * BYTES_PER_TOKEN,     # block_size is 64 tokens in this source
                  tier=Tier.CPU, representation=Representation.PLAINTEXT)
        for b, s in steps.items()
    ]
    traces = {b: AccessTrace(item_id=b, access_steps=s) for b, s in steps.items()}
    accesses.sort(key=lambda x: x[0])
    return Workload(items=items, accesses=accesses, traces=traces)


def enumerate_events(events):
    """Yield (timestep, conv_id, blocks) with timestep = global request order."""
    for t, (_wall, cid, blocks) in enumerate(events):
        yield t, cid, blocks


def load_kvct_with_meta(path_dir: str = DEFAULT_DIR, limit: int | None = None,
                        seed: int | None = None, offset: int = 0):
    """`load_kvct` plus the metadata a session-aware featurizer needs.

    Returns `(workload, meta)` where meta has:
        wall[timestep]      -> wall-clock seconds of that request
        conv_of[block_id]   -> owning conversation
        conv_steps[conv_id] -> timesteps at which that conversation issued requests

    Deriving features from THIS workload (and its `time_split`) is what keeps the
    feature instant and the placement instant identical. Building them from a separate
    event stream silently puts them on different scales -- the defect recorded in
    docs/claims_dependency.md §D.
    """
    events: list[tuple[float, str, list[str]]] = []
    files: list[str] = []
    for conv in _iter_conversations(path_dir, limit, seed, offset):
        cid = conv.get("id", "c")
        files.append(conv["_file"])
        for req in conv.get("requests", []):
            hs = req.get("hash_ids") or []
            if hs:
                events.append((float(req.get("t", 0.0)), cid, [f"{cid}:b{h}" for h in hs]))
    events.sort(key=lambda e: e[0])

    steps: dict[str, list[int]] = defaultdict(list)
    accesses: list[tuple[int, str]] = []
    wall: dict[int, float] = {}
    conv_of: dict[str, str] = {}
    conv_steps: dict[str, list[int]] = defaultdict(list)
    for t, (w, cid, blocks) in enumerate(events):
        wall[t] = w
        conv_steps[cid].append(t)
        for b in blocks:
            steps[b].append(t)
            accesses.append((t, b))
            conv_of.setdefault(b, cid)

    horizon = len(events) or 1
    items = [
        ItemState(item_id=b, access_freq=len(s) / horizon,
                  size_bytes=64 * BYTES_PER_TOKEN, tier=Tier.CPU,
                  representation=Representation.PLAINTEXT)
        for b, s in steps.items()
    ]
    traces = {b: AccessTrace(item_id=b, access_steps=s) for b, s in steps.items()}
    accesses.sort(key=lambda x: x[0])
    wl = Workload(items=items, accesses=accesses, traces=traces)
    return wl, {"wall": wall, "conv_of": conv_of, "conv_steps": dict(conv_steps), "files": files}


def wallclock_stats(path_dir: str = DEFAULT_DIR, limit: int | None = None) -> dict:
    """Think-time / session statistics this source exposes and Mooncake does not."""
    gaps: list[float] = []
    lengths: list[int] = []
    spans: list[float] = []
    for conv in _iter_conversations(path_dir, limit):
        ts = [float(r.get("t", 0.0)) for r in conv.get("requests", []) if r.get("hash_ids")]
        if len(ts) < 2:
            lengths.append(len(ts))
            continue
        ts.sort()
        gaps.extend(b - a for a, b in zip(ts, ts[1:]))
        lengths.append(len(ts))
        spans.append(ts[-1] - ts[0])
    gaps.sort()

    def pct(xs, p):
        return xs[int(p * (len(xs) - 1))] if xs else 0.0

    return {
        "conversations": len(lengths),
        "requests": sum(lengths),
        "median_requests_per_conv": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "think_time_p50_s": round(pct(gaps, 0.50), 2),
        "think_time_p90_s": round(pct(gaps, 0.90), 2),
        "median_session_span_s": round(sorted(spans)[len(spans) // 2], 2) if spans else 0.0,
    }
