# Data acquisition

None of the data this project measures is redistributed here. Each source is fetched from its
origin by the commands below. Two reasons: neither Mooncake nor LoCoMo states an explicit licence
for its data files, and shipping them would grow this repository from about 4 MB to 330 MB.

Total download is roughly 140 MB and takes a few minutes.

## Mooncake traces (required for every placement and tiering result)

Replayed production traces from a large LLM serving deployment, released with the Mooncake paper
(FAST '25). The repository is Apache-2.0; **the trace files carry no separate licence statement**,
and are described upstream as "anonymized request traces containing request arrival times, input and
output token counts, and remapped block hashes." Confirm current terms before redistributing.

```bash
mkdir -p data
curl -sL -o data/mooncake_toolagent_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/toolagent_trace.jsonl
curl -sL -o data/mooncake_conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/conversation_trace.jsonl
```

**Pinned** to Mooncake commit `3cca71daccf2` (the URLs above name it), so an upstream change cannot
alter these files silently; `data/SHA256SUMS` records each file's SHA-256 and `make data-check`
verifies them. (The URLs pointed at `main` until the 2026-10-16 audit, item M2.)

Expect 12,031 (conversation) and 23,608 (toolagent) request lines. Several results run both traces
**to exhaustion**, so a truncated download changes the numbers silently — `wc -l` them
(`make data-check` does).

**Block size is 512 tokens** — Mooncake's release README says so, and the data agrees:
`len(hash_ids) == ceil(input_length / 512)` for every one of the 35,639 production requests.
`stoa.mooncake.load_mooncake` enforces that identity per request and refuses a file that breaks it.
(This repository assumed 256 until 2026-10-02; placement results were invariant to the
correction, the calibration sweep was not — see `docs/claims_dependency.md` §AO.)

### The synthetic control

Mooncake's third FAST'25 workload is built from public datasets with Poisson arrivals. The paper
uses it as a **control**, never pooled with the production pair:

```bash
curl -sL -o data/mooncake_synthetic_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/synthetic_trace.jsonl
```

Expect 3,993 lines.

### Not used: `arxiv-trace/mooncake_trace.jsonl`

Mooncake also publishes an older single-file trace from its arXiv report. It is **not a fourth
workload**: at every one of its 23,608 positions it has the same `output_length` as the toolagent
trace, and the same number of blocks at 98.9% of them; only input lengths and timestamps were
re-derived. Counting it would count toolagent twice.

## LoCoMo (required for the placement-vs-retrieval and representation results)

Long-conversation QA benchmark (arXiv:2402.17753), from Snap Research. Downloads automatically on
first use, or fetch it directly:

```bash
python3 -c "from stoa.eval.locomo import load_locomo; load_locomo()"
```

10 dialogues, 1,986 questions. The repository has a `LICENSE.txt`; **check it before redistributing**.

## kv-cache-tester (needed for Table 2, Table 3 and §V-F)

Anonymized agentic-coding conversations from `github.com/callanjfox/kv-cache-tester`, pinned to upstream
commit `94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf` (2026-07-12). `data/kvct_SHA256SUMS` lists the 200 files
the paper uses with their SHA-256; `download` fetches them from that commit and verifies each one, and the
loader draws only from that list and raises on a missing or altered file.

```bash
python3 -c "from stoa.kvct import download; download(200)"
```

The paper uses this source in three places: Table 2's agentic column (`run_source_stats.py`), the
sampling study behind Table 3 and §V-E (`run_sampling_study.py`, a **negative** result: the quantity
has a spread an order of magnitude wider than any effect), and the eviction-signal inversion in §V-F
(`run_reactive_kvct.py`). `make reproduce-full` runs all three; `make verify` and `make reproduce-fast`
run without it.

## PrefixSim's production traces (optional: §V-K, the matching-rule audit)

Two traces released with "When Fancy Eviction Fails" (Liu, Yu, Yang 2026): `freeinference` and `chutes`, in
PrefixSim's lcp format, from the Hugging Face dataset `harvardMadsys/2026_prefixcache` at revision
`ba89a343a048073dda72617bb6087910da6214d0` (CC-BY-4.0). Not redistributed here. `scripts/run_prefixsim_audit.py
setup DIR` downloads them into `DIR` and checks their SHA-256 against the values pinned in the script.

## Alibaba Qwen Bailian KV traces (optional: the external replication cited in the paper's §V-C)

Four anonymized two-hour traces from `github.com/alibaba-edu/qwen-bailian-usagetraces-anon`
(Wang et al., "KVCache Cache in the Wild", USENIX ATC'25), 16-token blocks, Apache-2.0. Not
redistributed here. `python3 scripts/run_bailian.py --birthx` downloads them into `data/`, converts them
to the Mooncake schema the loader checks, and writes `experiments/bailian_*_birthx.json`; see notebook §AT.
The serving replay (§5.8, notebook §BM) turns a slice of `bailian_qwen_traceA_blksz_16.jsonl` into
token-id prompts with `scripts/serving_replay/make_workload.py`; the workloads are derived data and are
not committed either, but their SHA-256 digests are in `experiments/write_contract_a6000.json`.

## Derived caches (never committed, rebuilt on demand)

`data/emb_cache/` and `data/repr_cache.json` hold OpenAI embeddings and LLM rewrites of LoCoMo text.
Both are keyed by content and rebuild on a cache miss, so deleting them costs API calls, not
correctness. They are excluded from version control because they are derivatives of data this
repository does not redistribute.
