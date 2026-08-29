# Reproducibility package

Every number in the paper is produced by a script in this repository from public data. This file maps
each claim to the command that regenerates it, states what is and is not calibrated, and lists the
controls that are wired into the harness.

PVLDB EA&B submissions must link a package like this at submission time; the artifacts below are the
package.

## 1. Environment

CPU only — no GPU is required anywhere in this work. Every experiment trains against a tiering
simulator or a trace, never against an LLM.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # pytest
pip install -e ".[rl]"       # torch (CPU) + numpy — GNN/RL experiments only
pip install -e ".[eval]"     # httpx — LLM-backed QA only (needs an API key)
pytest -q                    # 163 tests, ~100 s; torch-gated tests skip cleanly without [rl]
```

Reference machine: 112 CPU cores, 502 GB RAM, Python 3.10, torch 2.13 (CPU), numpy 2.2.
The long-running items below are marked with wall-clock times observed there.

## 2. Data

Both trace sources are public and fetched by script. Neither requires credentials.

```bash
# Mooncake production traces (~7 MB) — Qin et al., FAST'25 (arXiv:2407.00079)
curl -sL -o data/mooncake_conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
curl -sL -o data/mooncake_toolagent_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl

# kv-cache-tester agentic traces (~130 MB for 200 conversations)
python3 -c "from stoa.kvct import download; download(200)"

# LoCoMo (auto-downloads on first use)
python3 -c "from stoa.eval.benchmarks import load; load('locomo')"
```

`data/` is gitignored; nothing here is redistributed.

## 3. Retracted claims

A five-reviewer simulation (2026-08-13) falsified this work's original central claim. The details are in
`docs/claims_dependency.md` §I; in short, three defects in our own experiment design produced it:

| retracted | why |
|---|---|
| "Mooncake omits request timing" | it does not; our loader discarded the `timestamp` field (fixed) |
| "AUC 0.43–0.64 vs 0.95, two regimes" | a trace-length artifact; at matched length both reach 0.81–0.93 |
| "learning captures 66–88% of headroom on agent traces" | reverses to −48% at 150 conversations; the 40/70-conversation samples were nested prefixes of a heterogeneous corpus |

Anything in `experiments/` predating that date should be read against §I before use. The surviving
result is the decoupling in §4 below.

## 3b. Environment, and what is hand-rolled

```bash
pip install -r requirements-lock.txt     # exact versions every paper number came from
```

`pyproject.toml` keeps loose lower bounds so the package stays installable; the lockfile is the
provenance record. The reference environment is Python 3.10 with torch 2.13.0+cpu, numpy 2.2.6,
httpx 0.28.1, pytest 9.1.1, matplotlib 3.10.9.

**scipy, scikit-learn and LightGBM are absent by design.** Fisher's exact test, exact McNemar and the
power simulations (`src/stoa/stats.py`), the gradient-boosted trees of the capacity ladder
(`src/stoa/gbdt.py`), and every cache policy (ARC, LeCaR, CACHEUS, LRB) are reimplementations with **no
library fallback path** — there is no `try: import scipy` anywhere. A committee member who has those
libraries installed therefore gets identical numbers; nothing in this artifact silently switches
implementation based on what is present. That removes a class of environment dependence, but it also
means the GBDT rung is not `HistGradientBoostingRegressor`: 200 depth-4 trees over 64 quantile bins with
no early stopping, against sklearn's leaf-wise 31-leaf default with an auto-enabled 10% holdout above
10k rows. The ladder trains on 30k–55k rows, so the two would differ materially.

**Determinism.** Every RNG is a local `random.Random(seed)`; `random.seed()` is called nowhere, so no
process-order-dependent global feeds a number. `scripts/run_capacity_ladder.py` pins
`torch.set_num_threads(1)`. The LLM-backed results (LoCoMo, representation axis) cannot be bit-reproduced
— they depend on a hosted model at `temperature=0`, which pins sampling but not the served weights.

**Cost.** `run_locomo_powered.py --judge` issues roughly 7,200 chat completions and
`run_representation_axis.py --judge` roughly 2,400, plus embeddings. At gpt-4o-mini list pricing that is
about **$3–5** and **$1** respectively. Everything else in this artifact is CPU-only and free.

## 3c. Before submitting: the artifact link

The repository is **private** while the paper is in revision. EA&B requires the reproducibility package
to be reachable at submission ("there are no excuses"), so before the abstract deadline:

1. `gh repo edit <owner>/stoa --visibility public --accept-visibility-change-consequences`
2. Put that URL in `\vldbavailabilityurl` in `paper/main.tex` and rebuild.
3. **Open the URL with no GitHub session** — `curl -sSo /dev/null -w '%{http_code}' <url>` must print
   `200`. Checking while logged in tests your access, not a reviewer's.

Step 3 is not paranoia. Every defect in `docs/claims_dependency.md` shares one shape: a report was
believed where an observation was available.

## 4. Claim → command → artifact

| Paper claim | Command | Artifact |
|---|---|---|
| Reuse distributions (Tab. 2, sample-dependent) | `python3 -c "from stoa.mooncake import *; print(trace_summary(load_mooncake('data/mooncake_toolagent_trace.jsonl')))"` | inline |
| Headroom (oracle gap) 81% at 1.5k requests → 87% at full trace, both traces | `sequential.reference_costs` at each length | `experiments/mooncake_length_sweep.json` |
| Binary reuse = 99.7% / 80.6% of headroom | `scripts/train_on_mooncake.py` | `experiments/mooncake_rq2.json` |
| **Decoupling grid**: AUC 0.57→0.93 while captured stays in [+0.8%, +2.9%], both traces run to exhaustion | `for n in 1500 6000 12031 23608; do python3 scripts/train_session_features.py --requests $n; done` | `experiments/mooncake_length_sweep.json`, `mooncake_session_rq2_full{12031,23608}.json` |
| **RETRACTED** — kvct think-time capture; sign reverses with sample size | `python3 scripts/train_kvct_placement.py --convs {40,70,100,150} --out experiments/kvct_placement_n{N}.json` | `kvct_placement_n150.json` (−48%) vs `kvct_placement_fixed.json` (+88%) |
| Reactive tiering on Mooncake, full traces (75–99% of Belady on toolagent, 40–94% on conversation) | `python3 scripts/run_reactive_real.py` (defaults to the full trace) | `experiments/reactive_real_full.json` |
| ARC meets the envelope at 5%/10% tiers only; adapts on 3.5–7.3% of misses | same run — `arc`, `arc_over_belady`, `arc_ghost_hit_rate_of_misses` columns | `experiments/reactive_real_full.json` |
| **LRB below ARC at all 8 operating points** | `python3 scripts/run_reactive_real.py --lrb` (~35 min) | `experiments/reactive_real_full_lrb.json` |
| LRB spans 55–66% of Belady across its hyperparameter grid; 78–87% of its training rows censored | `python3 scripts/run_lrb_sweep.py` (~25 min) | `experiments/lrb_sweep.json` |
| **Split-instant sensitivity**: linear ≈ GBDT to within 0.2 pts at all 20 configurations; captured in [+0.2%, +3.6%]; oracle gap 81→92% | `python3 scripts/run_split_sensitivity.py` | `experiments/split_sensitivity.json` |
| Working-set threshold on kvct (35% at a 1% tier → 99% at 10%) | `simulate_fast` sweep over `cache_frac`, 20 & 40 conversations | `experiments/reactive_kvct_sweep.json` |
| LoCoMo: retrieval dominates placement | `python3 scripts/run_locomo_baselines.py` (needs `OPENAI_API_KEY`) | `experiments/eval_locomo_leakfree.json` |
| **Capacity ladder**: linear / GBDT / MLP agree on placement to within 0.4 pts despite a 29-pt AUC spread | `python3 scripts/run_capacity_ladder.py` | `experiments/capacity_ladder.json` |
| **Cost model derived from hardware, swept over 3 attention shapes × 5 device classes: headroom 64–88%, hand-written table 81%, 6/15 hierarchies non-monotonic** | `python3 scripts/run_calibration_sensitivity.py` | `experiments/calibration_sensitivity.json` |
| LeCaR + CACHEUS on both full traces | `python3 scripts/run_reactive_real.py --lrb` | `experiments/reactive_real_full_lrb.json` |
| **LoCoMo at adequate scale**: 10 dialogues, 300 questions/arm, paired McNemar + per-dialogue sign test | `python3 scripts/run_locomo_powered.py --judge` (needs `OPENAI_API_KEY`, ~2.5 h) | `experiments/eval_locomo_powered.json` |
| **Representation axis measured** (plaintext / summary / extract at equal token budget) | `python3 scripts/run_representation_axis.py --judge` | `experiments/representation_axis.json` |
| LoCoMo null reported with its power: 0/18 comparisons survive Bonferroni; 246–710 questions/arm needed for the placement-vs-random contrast (33–347 for the others) vs 30 scored — and 758–2589 at the corrected threshold, see §S | `python3 scripts/run_locomo_power.py` (no LLM calls) | `experiments/locomo_power.json` |
| Cost-model sensitivity, uniform-multiplier (superseded — it varies scale, not ratios) | `python3 scripts/run_sensitivity.py` | `experiments/sensitivity.json` |
| kvct tier-size sweep (35% → 99% of Belady) | `run_online`/`simulate_fast` sweep, 20 & 40 conversations | `experiments/reactive_kvct_sweep.json` |
| Figures | `python3 scripts/make_figures.py` | `paper/figs/*.pdf` |
| **Every number above, checked against its artifact** | `python3 scripts/check_paper_numbers.py` (exits non-zero on drift) | — |

## 5. What is calibrated, and what is not

`docs/claims_dependency.md` classifies every claim:

- **§A cost-model independent** — reuse distributions, AUCs, hit rates. These are properties of the
  traces and would not move if the cost tables changed. The paper's conclusions rest here.
- **§B cost-model dependent** — anything expressed as weighted placement cost, including "headroom".
  The simulator's latency/token/promotion tables are illustrative placeholders; the calibration gate
  against a real vLLM+LMCache deployment has **not** been passed. Read these as shape, not magnitude.
- **§C not exercised** — the representation and timing axes. The traces record KV-block accesses only.
- **§D/§E/§F/§G** — invalidated, corrected, re-measured at full scale, and the sensitivity sweep.
  All kept deliberately, each with the defect or the change described.

## 6. Controls wired into the harness

Seven defects during this work produced plausible-but-wrong numbers. The checks that catch them are in the
code and the tests, not just in prose:

| Control | Where |
|---|---|
| Demand estimated only from held-out history questions | `memqa.split_history`, `MemQATask.leak_free`, `tests/test_leakage.py` |
| Policy features restricted to the observable prefix | `traces.prefix_stats` / `future_counts` (docstring states the rule) |
| Oracle must rank by the future, not inherit the heuristic's order | `sequential._prepare(order_by="future")` |
| Splits taken over time, never over access-event indices | `Workload.time_split` |
| One split instant shared by features and costs | `kvct.load_kvct_with_meta`, `place_with_beliefs_at` |
| Ranking metric handles ties by mid-rank | `scripts/train_session_features.py::auc` |
| A no-learning baseline of the same form as the learned one | `sequential.prefix_greedy_cost` |
| **(missing — the gap that cost us)** sweep every parameter fixed for convenience | not wired in; trace length, pool size, and tier size each produced a reversed claim |
| Optimized simulator pinned to the reference implementation | `eval/online.simulate_fast`, `tests/test_online_fast.py` (96/96 exact match) |
| Eviction tie rules stated explicitly (LFU, then LRU) | `eval/online._victim` / `key_of` — an implicit rule made two correct implementations disagree |

`pytest -q` runs all 163 tests, including the leakage guards and the fast/reference equivalence
check, in about 30 s.

## 7. Known limitations

- Cost tables are uncalibrated (see §4); no real-system deployment was performed. `scripts/run_sensitivity.py`
  bounds what this can cost: headroom stays in 65.6–93.7% across three orders of magnitude of perturbation.
- kv-cache-tester reactive results now sweep tier size over 20 and 40 pooled conversations (up to 7.9M
  accesses), made affordable by the heap-based simulator.
- Two trace sources, both KV-block level. No workload here exercises representation or timing choice.
- LLM-backed LoCoMo results depend on a third-party API and a substring/LLM-judge metric, not the
  benchmark's official scorer, and rest on n=30 questions — direction only, not a measurement.
- Conversation sampling is a sorted prefix (`kvct._iter_conversations`), so repeated "samples" are nested.
  The corpus is heterogeneous along that index; randomized sampling with intervals is required and absent.
- The `.env` used for the LLM experiments has been deleted; the exposed key should be rotated.
