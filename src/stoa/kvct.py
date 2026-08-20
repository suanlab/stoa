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
"""
from __future__ import annotations

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

BASE_URL = "https://raw.githubusercontent.com/callanjfox/kv-cache-tester/master/traces/"
DEFAULT_DIR = "data/kvct"
BYTES_PER_TOKEN = 2 * 2 * 32 * 128 * 2 // 1024      # illustrative; calibrate in M0-3


def download(n: int = 200, out_dir: str = DEFAULT_DIR, workers: int = 16) -> int:
    """Fetch the first `n` conversation traces. Returns how many are present."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def one(i: int) -> bool:
        name = f"trace_{i:04d}.json"
        path = Path(out_dir) / name
        if path.exists() and path.stat().st_size > 100:
            return True
        try:
            urllib.request.urlretrieve(BASE_URL + name, path)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(one, range(1, n + 1)))


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
    files = sorted(Path(path_dir).glob("trace_*.json"))
    files = files[offset:]
    if seed is not None:
        rng = random.Random(seed)
        files = rng.sample(files, min(limit or len(files), len(files)))
    elif limit is not None:
        files = files[:limit]
    for f in files:
        try:
            yield json.loads(f.read_text())
        except Exception:
            continue                                  # skip a truncated download


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
    for conv in _iter_conversations(path_dir, limit, seed, offset):
        cid = conv.get("id", "c")
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
    return wl, {"wall": wall, "conv_of": conv_of, "conv_steps": dict(conv_steps)}


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
