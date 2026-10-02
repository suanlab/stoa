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
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
curl -sL -o data/mooncake_conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
```

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
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl
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

## kv-cache-tester (optional — used only for the cautionary sampling study)

Anonymized agentic-coding conversations from `github.com/callanjfox/kv-cache-tester`.

```bash
python3 -c "from stoa.kvct import download; download(200)"
```

This source supports exactly one result in the paper, and that result is a **negative** one: the
quantity we tried to measure on it has a spread an order of magnitude wider than any effect
(`docs/claims_dependency.md` §J). Everything else runs without it.

## Derived caches (never committed, rebuilt on demand)

`data/emb_cache/` and `data/repr_cache.json` hold OpenAI embeddings and LLM rewrites of LoCoMo text.
Both are keyed by content and rebuild on a cache miss, so deleting them costs API calls, not
correctness. They are excluded from version control because they are derivatives of data this
repository does not redistribute.
