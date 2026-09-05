# Which claims depend on the (uncalibrated) cost model?

> **Why this file exists.** Our simulator's cost tables (`simulator._TIER_READ_MS`,
> `_REPR_TOKENS`, `_PROMOTE_COST`) are illustrative placeholders; the M0-3 calibration gate against a
> real vLLM+LMCache deployment has not been passed. A reviewer is right to discount anything that
> rests on them. This file separates the results that do from the results that do not, so the paper
> can lead with the robust ones and quarantine the rest.
>
> We also learned the hard way that a mis-specified cost model is not a harmless approximation: under
> ours, the per-item optimal action is identical for every item with any future reuse, which collapses
> a three-axis decision into a one-dimensional sort and makes every policy look equivalent.

## A. Cost-model INDEPENDENT (safe to lead with)

These are properties of the traces or of hit/miss counting. They would not change if the cost tables
were replaced tomorrow.

| Claim | Metric | Artifact |
|---|---|---|
| Reuse distribution is extremely bimodal (Mooncake: ~78% of blocks touched once; a few touched thousands of times) | block-level counts | `mooncake.trace_summary` |
| The two production sources occupy **different reuse regimes** (Mooncake ~78% one-time, mean reuse 1.9; kv-cache-tester ~7.5% one-time, mean reuse 25.6) | block-level counts | `kvct.wallclock_stats`, `mooncake.trace_summary` |
| Agent traces re-send their prefix, so a request touches ~2.5k blocks vs ~17 in Mooncake | blocks per request | both adapters |
| Binary reuse is barely predictable from block/session features (AUC 0.43–0.64, no transfer) | AUC (ranking only) | `scripts/train_session_features.py` |
| Reactive tiering reaches 82–93% of Belady's **hit rate** on the agent trace, 37–45% on conversation | hit rate | `scripts/run_reactive_real.py` |
| Belady > H2O/LFU > LRU ordering | hit rate | `eval/online.py` |
| kv-cache-tester exposes wall-clock think time (p50 20 s, p90 131 s, median session span 42 min) | timestamps | `kvct.wallclock_stats` |

## B. Cost-model DEPENDENT (report as shape, with a sensitivity check)

These use `weighted_cost` over the placeholder tables. Their *direction* is meaningful; their
magnitudes are not evidence.

| Claim | Why it depends | Mitigation |
|---|---|---|
| "~80% headroom between the frequency heuristic and the oracle" | headroom is a ratio of weighted placement costs | report alongside the hit-rate version; state as shape |
| "The binary oracle captures 99.7% of the headroom" | decomposition of the same weighted cost | re-run under perturbed cost tables (sensitivity) |
| "A learned policy captures ≈0% of the headroom" | same | the AUC result (§A) carries the same conclusion without costs |
| Budget-conditioned frontier dollar values | promotion costs are placeholders | present the frontier's *shape*, not the dollar axis |
| Any claim about the representation or timing axes | those axes are exercised only through the cost tables; the traces contain no representation signal at all | scope explicitly: this work characterizes the **tier** axis |

## C. Not exercised by the current data at all

The traces record KV-block accesses. They say nothing about representation choice (plaintext vs
vector vs latent) or consolidation timing. Every result about those two axes is therefore simulator
-internal and should be presented as formulation, not measurement. Validating them needs either a
system that actually performs the promotions or a workload that records them.

## Practical rule for the paper

Lead §5 with the §A results, and mark every §B number in the text as resting on an uncalibrated
cost model. If a claim in §B is load-bearing for a conclusion, check whether the §A version of that
conclusion exists — for our two main conclusions (reuse is unpredictable; reaction beats prediction)
it does.

---

## D. INVALIDATED — do not cite (2026-08-11)

| Claim | Artifact | Defect |
|---|---|---|
| kv-cache-tester: learning captures +15.4% / −29% / −16.5% of placement headroom | `experiments/kvct_placement*.json` | **Feature/placement split mismatch.** Features and labels were built from an event stream containing only the evaluation conversations (split at index 357 of 715), while placement costs were billed on a workload pooling *all* conversations (split at timestep 2107 of 4215). Beliefs fitted at one instant were applied to costs at another. `scripts/train_kvct_placement.py` now refuses to run until features are derived from the same `Workload` used for placement, as `train_on_mooncake.py` does. |

Unaffected by this defect: every AUC number from `train_thinktime.py` (self-consistent within one
stream), all descriptive trace statistics, the Mooncake placement results (built from a single
`Workload`), and the reactive-tiering results.

**Pattern worth noting.** This is the sixth measurement defect found in this project, and the fifth
that produced a *plausible* number rather than an obvious error. All six share one shape: two
quantities that must be defined on the same reference frame — same split instant, same ordering, same
tie handling, same question set — were silently defined on different ones. The cheap control is to
assert the frame explicitly (e.g. compare split indices, compare id orderings) rather than trust that
two code paths agree.

## E. CORRECTED — single-operating-point extrapolation (2026-08-12)

An earlier draft claimed "reactive tiering collapses to 21--25\% of Belady on agent traces", making
reaction and prediction look cleanly complementary across workloads. That figure came from a **single
cache size (2%)**. Sweeping the size shows the claim is false as stated:

| kv-cache-tester | 2% hot tier | 5% hot tier |
|---|---|---|
| reactive / Belady | 25--33% | **94%** |

The real effect is a working-set threshold: an agent request touches ~2.5k blocks, so a 2% tier cannot
hold it and thrashes, while a 5% tier can. Reaction is *not* inherently weak on agent workloads.

Secondary finding worth keeping: on agent traces **LRU beats H2O/LFU decisively** (0.884 vs 0.228 hit
rate at 5%), the reverse of Mooncake. Agent loops re-send a contiguous recent prefix, so recency is the
right signal there and frequency is the right signal on general serving traces.

**Lesson (the seventh defect, and the first of its kind here):** the previous six were reference-frame
mismatches; this one was extrapolating a curve from one point. Any claim of the form "mechanism X fails
on workload Y" needs the sweep, not a sample.

## F. RE-MEASURED AT FULL SCALE (2026-08-12)

Reactive-tiering numbers were first computed on an 800-request subsample. Rerunning on the full
Mooncake traces (23.6k requests, 183k blocks) changed both the values and the trend:

| reactive / Belady | 800-request subsample | full trace |
|---|---|---|
| toolagent, 2% / 5% / 10% tier | 93% / 89% / 82% | **75% / 81% / 92%** |
| conversation, 2% / 5% / 10% tier | 45% / 37% / 41% | **40% / 53% / 76%** |

The subsample suggested reaction gets *worse* as the tier grows, which is backwards; at full scale it
improves monotonically, as it should. The paper reports the full-trace values and states the discrepancy.

This was made affordable by `eval/online.simulate_fast`, which replaces the per-eviction linear scan with
lazily-invalidated heaps (53x faster on a 3k-request slice). It is pinned to the reference implementation
by `tests/test_online_fast.py`, which also fixed an under-specified LFU tie rule: the reference broke ties
by set-iteration order and the heap version by insertion order, so two correct implementations disagreed
by ~0.5% of hits until the tie-break (LFU, then LRU) was written down explicitly in both.

## G. SENSITIVITY — how far §B claims survive perturbed cost tables (2026-08-12)

`scripts/run_sensitivity.py` perturbs the promotion, latency, and token tables over three orders of
magnitude on the Mooncake toolagent trace:

| perturbation | headroom |
|---|---|
| baseline | 80.4% |
| promotion ×100 / ×1000 | 75.4% / 65.6% |
| latency ×10 / ×0.1 | 75.6% / 92.0% |
| tokens ×10 / ×0.1 | 93.7% / 75.3% |
| promotion ×100 + tokens ×10 | 83.9% |
| **range** | **65.6–93.7%** |

The headroom magnitude is cost-dependent, but the qualitative claim — a frequency heuristic leaves most
of the achievable gain unrealized — holds under every setting tested. §B claims should be read as this
range, not as points. This does not substitute for calibration; it bounds what the uncalibrated framing
can be wrong by.

## H. PROCESS NOTE — a silent no-op edit (2026-08-12)

The sensitivity subsection was reported as inserted when it had not been: the edit script targeted a
subsection heading (`The headroom is one binary question`) that an earlier rewrite had already removed,
so `str.replace` matched nothing and the script's unconditional success message was wrong. Caught by
listing the section structure afterwards.

Same family as the seven measurement defects: a claim believed on the strength of a *report* rather than
an *observation*. The fix is the same shape too — assert the anchor exists before writing, and grep for
the inserted content afterwards, rather than trusting a print statement.

---

## I. RETRACTED — the paper's central positive claim (2026-08-13)

A five-reviewer simulation, plus our own verification, falsified the "timing makes it learnable" thesis.
Three independent defects produced it, all in our experiment design rather than in the workloads:

**I.1 — Mooncake does record request timing; our loader discarded it.**
`load_mooncake` used the line index as the timestep and never read the `timestamp` field, although the
module docstring listed it. Both traces span ~59 minutes at ~3 s resolution (1,180 distinct values).
Every statement of the form "Mooncake omits request timing" was false. Fixed 2026-08-13; the loader now
returns wall-clock times via `with_wallclock=True`. A reviewer added think-time features to Mooncake
using the field and measured **no effect at full trace length** (Δ AUC −0.001 / +0.000).

**I.2 — The AUC contrast (0.43–0.64 vs 0.95) was a trace-length artifact.**
Mooncake experiments used 1,200–2,000 of 23,608 requests; kv-cache-tester used every request of the
sampled conversations (~300× more accesses). At matched length Mooncake reaches AUC 0.81–0.93. There is
no "two regimes" contrast.

**I.3 — The headline "+66 to +88% of headroom captured" reverses sign with sample size.**
Re-running our own unmodified script:

| conversations | think-time headroom captured (cls / regr) |
|---|---|
| 40 (reported) | +66.2% / +87.6% |
| 70 (reported) | +86.6% / +74.5% |
| 100 | +27.6% / −8.5% |
| **150 (verified here)** | **−38.3% / −48.3%** |

At 150 conversations the learned policy is *worse than the no-learning frequency heuristic*. Our claim
that "only the think-time row reproduces across samples" was exactly backwards.

Root cause: `kvct._iter_conversations` takes `sorted(files)[:limit]`, so the 40- and 70-conversation runs
are **nested prefixes**, not independent samples — and the corpus is systematically heterogeneous along
file index (mean requests/conversation 140 → 37; one-time fraction 0.127 → 0.044 from files 1-20 to
71-150). Verified independently here.

**I.4 — Related mechanism claims that fall with it.**
- The working-set threshold is governed by the *aggregate concurrent working set*, not the per-request
  footprint: holding tier size fixed in absolute blocks and raising concurrency still collapses the ratio.
- "Reactive tiering reaches X% of Belady" is a per-operating-point argmax over LRU, LFU and a
  never-evicting `static` policy. No single deployable policy achieves the reported band.
- §5.6's "placement needs calibrated magnitude" is wrong for our cost model: rank-normalizing every
  belief (order preserved, magnitude destroyed) changes +66.2/+87.6% to +63.9/+86.2%. The effect is ~97%
  ordering. The −129.3% cell in Table 3 collapses to the −5.2% floor and is a belief-scale artifact.

**What survives** (verified at full scale by two reviewers and here): reuse-ranking quality does not
convert into placement benefit — capture stays at 0.5–2.8% across an AUC sweep of 0.57→0.93; a frequency
heuristic leaves most of the headroom on both sources; and the defect taxonomy itself.

**Process lesson.** The eight defects we found ourselves were all *internal-consistency* failures. None
of our checks questioned the free parameters of our own experiment design — trace length, conversation
count, which fields the loader reads. Every claim should be accompanied by a sweep over the parameter
that was chosen for convenience.

## J. THE CONTROL THAT WAS MISSING (2026-08-13)

`scripts/run_sampling_study.py` draws several RANDOM conversation subsets per size (the loader now takes
a `seed`) and reports mean ± 95% CI for the same pipeline that produced the retracted claim:

| conversations | headroom captured, 4 random draws |
|---|---|
| 20 | **−541% ± 660** |
| 40 | **−67% ± 126** |

The interval is an order of magnitude wider than the effect. The original +66–88% was one draw from this
distribution, taken as a sorted prefix. So the quantity was not merely mis-estimated — it was never
measurable under this protocol, and no amount of care in the *analysis* would have revealed that. Only
re-drawing the sample did.

Cost of the control: about five minutes of CPU. It is now in the harness.

Two further corrections made in the same pass:
- **Tier hierarchy was non-monotonic.** `REMOTE_RDMA` was both faster (6 ms vs 12 ms) and larger
  (unbounded vs 1.0×) than `DISK`, so no cost-minimizing policy would ever choose disk and the "four
  tiers" were three. Capacities are now monotone in latency (`disk` unbounded, `remote_rdma` 0.6×), and
  `tests/test_rl.py::test_tier_hierarchy_is_monotonic` pins the invariant. The existing regression test
  caught the change immediately, which is the behaviour we wanted.
- **The `h2o` policy is LFU.** H2O (arXiv:2306.14048) evicts tokens within a sequence by accumulated
  attention score, which a block-hash trace cannot express. The alias is kept for artifact compatibility;
  the docstring and the paper now say LFU.

## K. CONTROLS ADDED 2026-08-13 (post-review)

Three gaps the reviewer simulation named were closed. All are cost-model **independent** (§A): they turn
on AUC and hit rate, not on the placeholder cost tables.

### K.1 Model capacity is not the bottleneck (`scripts/run_capacity_ladder.py`)

The decoupling claim was open to "you used too weak a model." A ladder holding features, split instant,
cross-trace pairing, placement routine and billing fixed, varying only capacity, at FULL trace length:

| model | AUC conv / tool | captured conv / tool |
|---|---|---|
| constant (no learning) | 0.500 / 0.500 | −10.3% / −24.3% |
| log(prefix count) alone | 0.639 / 0.709 | −3.5% / −2.2% |
| logistic regression | 0.812 / 0.925 | +2.3% / +2.1% |
| GBDT (200×depth 4) | 0.804 / 0.912 | +2.3% / +1.9% |
| MLP (2×64) | 0.820 / 0.931 | +2.3% / +2.1% |
| **true future counts** | 1.000 | **100%** |

Three model classes agree on placement to within **0.4 points**. The capacity is genuinely there and
genuinely used — fitted and scored *within* one trace on disjoint blocks, GBDT reaches AUC 0.934/0.966
against the linear model's 0.814/0.929 — it simply does not convert. In-trace numbers are reported for
ranking only: scoring placement in-trace would credit a model for placing the blocks it was fitted on.

A length caveat worth recording: at a 6k-request prefix the cross-trace GBDT AUC is 0.66, below the
linear model's 0.77; at full length it recovers to 0.80. Reading the capacity comparison at 6k would have
produced the wrong conclusion ("trees don't transfer") for the right-sounding reason. Another instance of
§I's lesson.

### K.2 ARC evaluated, and why it only half-works here (`scripts/run_reactive_real.py`)

ARC (Megiddo & Modha, FAST'03) is now implemented (`stoa.eval.online.simulate_arc`, `tests/test_arc.py`)
and run on both full traces. As a single deployable policy it reaches 71/84/92/96% of Belady on toolagent
and 29/58/78/87% on conversation at 2/5/10/20% tiers — **beating the best pure rule at 5% and 10% on both
traces**, so the previously-reported "envelope" is deployable at those operating points. At the extremes
it loses: LFU by ~10 points at 2%, LRU by ~7 points at 20%.

Measured mechanism: ARC updates its target split `p` only on a **ghost hit**, and 76–78% of blocks in
these traces are never reused, so most evictions are correct and emit no signal. ARC receives an
adaptation signal on only **3.5–7.3% of misses**. At a 2% tier it drives `p` to 90% recency on both
traces while the winning rule there is frequency — sparse *and* directionally wrong. This is a property
of KV workloads, not of ARC.

Recorded as new fields in `experiments/reactive_real_full.json`: `arc`, `arc_over_belady`,
`arc_ghost_hit_rate_of_misses`, `arc_final_p_frac`, `winner`.

### K.3 Trace length swept to exhaustion (`experiments/mooncake_length_sweep.json`)

The §5.2 table previously mixed lengths (6,000 for one trace, 23,608 for the other) — the free-parameter
defect the paper itself catalogs, committed in the table stating the headline. Both traces are now run at
1,500 / 6,000 / 12,031 / 23,608 requests, i.e. to exhaustion (conversation ends at 12,031):

- AUC spans 0.566 → 0.925 (36 points).
- Captured headroom stays in **[+0.8%, +2.9%]** and does **not trend with AUC** — its maximum is
  mid-sweep, not at the best ranker.
- The oracle gap itself moves with length: 81.0% at 1,500 requests → 87.2% at full. The paper's range is
  corrected from "82–87%" to **81–87%**.

### Artifact naming hazard fixed

`experiments/reactive_real.json` and `reactive_real_full.json` had diverged silently, with
`make_figures.py` preferring the latter. The run script now derives its output name from the scale
(`_full` vs `_sub{N}`), so a subsample cannot overwrite the full-trace artifact the paper reads.

## L. LRB EVALUATED (2026-08-14)

The one learned-cache baseline we had cited without running. `src/stoa/lrb.py` reimplements the published
mechanism (GBDT on log time-to-next-request over inter-access deltas and exponentially decayed counters,
sampled eviction, online retraining, right-censored labels); `scripts/run_reactive_real.py --lrb` runs it
on both full traces and `scripts/run_lrb_sweep.py` varies its knobs.

**Result: LRB is below ARC at every one of the eight operating points, and below LRU at most.**

| tier | conversation: LRB / ARC / best | toolagent: LRB / ARC / best |
|---|---|---|
| 2% | 27.9% / 29.4% / 39.8% | 70.2% / 70.7% / 74.9% |
| 5% | 44.3% / 57.9% / 57.9% | 76.5% / 83.6% / 83.6% |
| 10% | 64.0% / 77.5% / 77.5% | 84.7% / 92.3% / 92.3% |
| 20% | 78.2% / 87.3% / 94.0% | 91.7% / 96.2% / 99.3% |

(all as a fraction of Belady's hit rate)

**This is not a tuning artifact.** Sweeping memory window ∈ {10k, 50k, 200k}, sample size ∈ {32, 64,
128} and ensemble size ∈ {30, 100} at a 10% tier gives **55.0–65.7%** of Belady on conversation (ARC:
77.5%) and **80.9–85.4%** on toolagent (ARC: 92.3%). Every one of the twelve grid points is below ARC on
its trace. Widening the memory window *lowers* the score on both, even though it cuts the censored
fraction — it also collects far fewer rows (4.2M vs 10.5M on conversation).

**Measured mechanism.** 77–83% of LRB's training rows are right-censored at the default window (86–89% at a shorter one) — the
sampled block was never re-requested inside it, so the row says "not within the window" and nothing about
when. This is the same starvation that limits ARC (§K.2), seen from the learning side rather than the
adaptation side: a block population that is 76–78% singletons carries little information about *when*
anything returns.

**Two caveats stated in the paper.** It is our reimplementation, not the authors' artifact. And both
sources use uniform-size blocks, so byte miss ratio collapses to object miss ratio and LRB loses one of
its structural advantages over LRU. The result is a statement about this workload, not about LRB.

### L.1 A frame bug caught by the same check the paper prescribes

The first version computed a block's features at *request* time and labelled them with the gap to the
next request. Every training row therefore carried recency 0, while at eviction time recency is whatever
has elapsed — the model was fitted on a distribution it never sees in deployment. LRB scored **15 points
below LRU**. Recording features at eviction-sampling time (which is what the paper specifies) fixed it:
66.9% → 84.7% on a synthetic locality workload. `tests/test_lrb.py` pins the invariant, and
`test_learning_beats_its_own_lru_warm_up` is the no-learning control — the same policy with
`train_interval` past the trace end, so only the model is removed.

## M. THE LAST TWO FREE PARAMETERS SWEPT (2026-08-14)

§I's lesson was that every claim needs a sweep over the parameter chosen for convenience. Two remained.

**Trace length** — swept to exhaustion (§K.3).

**Split instant** — `scripts/run_split_sensitivity.py` sweeps it over {0.3, 0.4, 0.5, 0.6, 0.7}, feeding
one variable to both feature construction and cost billing. 20 configurations (5 splits × 2 traces × 2
model classes):

- Linear and GBDT agree to within **0.2 points at every configuration** — the capacity result of §K.1 is
  not an accident of the 0.5 split.
- Captured headroom stays in **[+0.2%, +3.6%]**.
- The oracle gap itself runs **81% (split 0.3) → 92% (split 0.7)**: a later split leaves less future to
  get wrong. So "the gap is 81–87%" is the *length* sweep at split 0.5; across splits it is 81–92%. The
  range must always be quoted with the parameter it was swept over.

## N. LoCoMo: A NULL REPORTED WITH ITS POWER (2026-08-14)

`scripts/run_locomo_power.py` + `src/stoa/stats.py` (exact Fisher, no scipy here; validated against
R's `fisher.test` in `tests/test_stats.py`). No new LLM calls — this re-analyses artifacts we had.

The paper previously said the LoCoMo arms were "not statistically separable" and moved on. That is a
weaker and less honest statement than the data supports, in both directions:

**1. Nothing survives correction.** Nine pairwise comparisons per run; Bonferroni threshold p < 0.0056.
**Zero survive in either run.** The smallest raw p we observe anywhere is 0.007. The retrieval-vs-placement
contrast reaches raw p = 0.010 and 0.039 at the two tighter budgets, which is suggestive and nothing more.

**2. The between-run reversal is NOT sampling noise.** Simulating two runs of n=16 and n=30 at the
leak-free run's own rates, P(opposite ranking) = 0.007 / 0.028 / 0.204 at the three budgets. So the
reversal cannot be waved away as small-n jitter. What differs between the runs is *which dialogues were
drawn* — two versus three, from a corpus of ten. **This is §I's failure in a different corpus**: a small
draw from a heterogeneous population, reported as though the population were homogeneous.

**3. The required sample size is computable, and we were far below it.** For 80% power at the observed
effects: placement vs random needs **246–710 questions per arm**; retrieval vs placement needs 33–196.
We scored **30**. LoCoMo releases **1,986** questions across ten dialogues, so the experiment is entirely
runnable at the scale it needs — we simply did not run it there.

The paper now reports the scoping argument on its merits and leaves the measurement open **with its
sample size specified**. New protocol rule: a null result is only reportable alongside the n that would
have resolved it, because "we could not tell" and "there is nothing there" are different claims.

## O. LeCaR AND CACHEUS EVALUATED (2026-08-14)

`src/stoa/experts.py`, `tests/test_experts.py`. These complete the adaptive-replacement family and test a
prediction the paper makes rather than asserting it: because LeCaR and CACHEUS read the same ghost-hit
channel ARC does, and only 3.5–7.3% of misses carry that signal here (§K.2), they should land near ARC
rather than near Belady.

### O.1 A bug that would have confirmed the prediction for the wrong reason

The first SR-LRU always evicted from the scan region R. With no bound on R's target size and no demotion
from S, S grows without limit until the policy is effectively a fill-once cache. It scored **6 points
below plain LRU** on a singleton-heavy workload — which is exactly the direction the paper predicts, and
would have been reported as confirmation.

`tests/test_experts.py::test_cacheus_scan_region_helps_under_heavy_one_time_traffic` caught it. The fix
is the published design: an adaptive target size for R (moved by history hits, ARC-style) and a
second-chance demotion of S's LRU into R before evicting. CACHEUS went 34.5% → 40.9% on that workload,
now above LRU's 40.4%.

The general lesson, and it is the sharpest one in this document: **a baseline that fails in the direction
your paper predicts is the one you must debug hardest.** Every other bug in this file was caught because
a result looked too good.

Both are reimplementations from the papers, not the authors' artifacts; CR-LFU is approximated by LFU
with an LRU tie-break, stated in the module docstring and in the paper.

## P. THE COST MODEL, DERIVED RATHER THAN ASSERTED (2026-08-14)

`src/stoa/calibration.py`, `scripts/run_calibration_sensitivity.py`, `tests/test_calibration.py`.

### P.1 The sensitivity analysis we already had was testing the wrong parameter

`run_sensitivity.py` multiplies every tier latency by a common factor. That moves the absolute scale and
leaves the **ratios between tiers untouched** — and the ratios are what a placement policy responds to.
An item goes to GPU because GPU is 10× faster, not because it is 0.2 ms fast. So the reported 65.6–93.7%
band was a sweep over a parameter the conclusion is largely insensitive to. This is the free-parameter
defect of §I, found in our own cost model, by us, after the paper already catalogued the pattern.

### P.2 What replaces it

Latency is now derived: `read_ms(tier) = overhead(tier) + kv_bytes_per_block / bandwidth(tier)`, with

    kv_bytes_per_block = 2 (K,V) × layers × kv_heads × head_dim × dtype_bytes × block_tokens

The second line is **exact arithmetic** over a model's attention shape — nothing to estimate. The
bandwidth and overhead are **declared assumptions about a device class, not measurements**; this machine
has no GPU and the live-deployment calibration gate (§B) is still open.

Sweeping 3 attention shapes × 5 device classes:

| quantity | value |
|---|---|
| headroom across 15 derived cost models | **64.2 – 87.8%** |
| hand-written table's headroom | 81.1% (inside the range) |
| hand-written GPU:CPU latency ratio | 1 : 10 |
| derived GPU:CPU ratio | 1 : 80 (balanced) to 1 : 250 (narrow host link) |
| device classes producing a non-monotonic hierarchy | **6 of 15** |

**The headroom conclusion survives** — 81% sits inside 64–88%. Two things about the table do not:

1. **The GPU:CPU ratio was off by an order of magnitude**, in the ratio the policy is most sensitive to.
2. **6 of 15 device classes reorder the hierarchy** so latency order and capacity order disagree. On a
   constrained host link the CPU tier becomes slower than the remote one. `_TIER_CAPACITY_FRACTION` is a
   fixed table and cannot express those machines at all — a limitation of the simulator, not of the
   device classes. This is the same non-monotonicity `tests/test_rl.py::test_tier_hierarchy_is_monotonic`
   was added to catch, arriving this time from realistic hardware rather than from a typo.

Block size (`block_tokens`) multiplies every tier's transfer term equally, so it moves scale and not
ratios — pinned by `test_block_size_moves_scale_but_not_ratios`. That is why the sweep varies device
class instead, and it is the distinction the old analysis missed.

## Q. LoCoMo, RE-RUN AT THE SIZE THE POWER ANALYSIS DEMANDED (2026-08-15)

§N established that the LoCoMo experiment resolved nothing and computed what it would take. With an API
key available, we ran it: **all 10 released dialogues, 30 held-out questions each, 300 scored questions
per arm, LLM-judged** (`scripts/run_locomo_powered.py`).

Two design changes, both consequences of §N:

1. **All ten dialogues.** Between-dialogue heterogeneity produced the reversal; drawing every dialogue
   removes it as a variable instead of hoping it averages out.
2. **Paired tests.** Every arm answers the *same* questions, so the comparison is paired: McNemar on the
   discordant pairs (`stats.mcnemar_exact`), plus a per-dialogue sign test so no result can rest on one
   dialogue. Fisher on independent samples throws away the pairing and is needlessly conservative.

| budget | demand | random | centrality | retrieval |
|---|---|---|---|---|
| 10% | 12.0% | 6.7% | 1.7% | **51.7%** |
| 25% | 16.7% | 11.3% | 6.3% | **52.7%** |
| 50% | 29.3% | 25.3% | 16.3% | **55.3%** |

**Six of nine comparisons now survive Bonferroni, where none did before.** Three distinct findings:

- **Retrieval dominates placement decisively.** Ahead at every budget, 4× at the tightest, p < 1e-15,
  and the same direction in **10 of 10 dialogues** (sign test p = 0.002). The scoping argument is now
  *measured*, not argued.
- **The placement signal is real in its own regime.** Demand beats embedding centrality at every budget
  (p ≤ 1.2e-4, 8–10 of 10 dialogues). This is a **positive** result the paper did not previously have.
- **Demand vs random is still unresolved**: p = 0.029 / 0.068 / 0.256, right direction throughout. 80%
  power needs **486–667 questions per arm** against the 300 run. Reported as open, with its n attached.

### Q.1 Two bugs the run exposed

- **`mcnemar_exact` overflowed** at large n: `sum(comb(n,i)) / 2**n` raises `OverflowError` past n≈1000,
  reachable both from a big discordant count and from the power simulation's upper bracket. Now computed
  in log space. The artifact had already been written when it fired, so no results were lost — but a
  crash after the expensive part is luck, not design.
- **`stoa.llm` had no retries.** The representation-axis run died on a single `httpx.ReadTimeout` two
  dialogues in and discarded ~1,600 paid compression calls. `complete()` and `embed()` now retry
  transient failures (timeout, transport, 429, 5xx) with exponential backoff and raise immediately on a
  non-429 4xx, and the representation script checkpoints its compression cache to disk after every
  dialogue.

## R. THE REPRESENTATION AXIS, FINALLY EXERCISED (2026-08-15)

Every experiment before this fixed representation at plaintext, which means the formulation's central
*coupling* claim — that representation and tier must be decided together because compression trades
tokens against utility — was asserted and never measured. `src/stoa/eval/representation.py` +
`scripts/run_representation_axis.py` measure the utility half on LoCoMo: same fact ids, same questions,
same gold answers, three representations (plaintext / LLM summary / key-value extract) differing only in
text and token cost.

**Critical frame requirement**: budgets are a fraction of the **plaintext** total for every mode. Sizing
each mode's budget from its own total would silently hand compressed representations a larger effective
allowance — the same reference-frame error this document keeps recording, and it would guarantee the
result we are testing for.

Results pending the run; whichever way it lands is reportable. A single winner at every budget weakens
the coupling argument (representation would not need to be per item); a crossing as the budget moves
confirms it and names the operating point a policy would have to learn.

### R.1 Result: the direction points against our own coupling claim

| budget | plaintext | summary | extract |
|---|---|---|---|
| 5% | 5.0% | **11.0%** | 5.0% |
| 10% | 10.0% | **15.0%** | 7.0% |
| 25% | 19.0% | **26.0%** | 21.0% |
| 50% | 32.0% | **38.0%** | 22.0% |

100 questions per cell, LLM-judged. Compression ratios: summary 0.58×, extract 0.19×.

**The trade-off is real** — at 0.19× the extract loses ten points to plaintext at the loosest budget, so
compressing far enough does destroy utility. **But the optimum does not move**: summary leads at all four
budgets, with discordant pairs favouring it every time (6–0, 5–0, 11–4, 19–13). No crossing.

If that survives at adequate n, representation on this workload need **not** be re-decided per operating
point — a weaker coupling than `docs/stoa_design.md` §2 and the paper's §3 assume. We wrote the script to
report either outcome before seeing this one.

**It does not yet survive.** 0 of 8 comparisons pass correction (best raw p = 0.031 vs a 0.00625
threshold); resolving summary-vs-plaintext needs **129–718 questions per cell** against the 100 run. By
the rule §N established, this is a direction with its n attached, not a result.

Two design points worth keeping: budgets are a fraction of the **plaintext** total for every mode (sizing
each from its own total would hand compressed forms a bigger allowance and manufacture the finding), and
the comparison is **paired** because all three modes answer the same questions.

## S. REVIEW SIMULATION ROUND 2 (2026-08-16) — WHAT IT FOUND

Five reviewers were commissioned. Four were cut off by an API quota; one (adversarial/reject) completed
and one (ML-for-systems) confirmed a single finding before stopping. Both of their substantive findings
were independently verified here and are recorded below with the fixes.

### S.1 THE ACTION SPACE IS DEGENERATE — the deepest finding, and it re-scopes the headline

Reviewer C: *"CONFIRMED: action choice saturates at belief ≥ 0.1."* Verified:

- Sweeping the believed reuse count from 0 to 1000, the per-item `argmin` over the 50 legal actions
  yields **2 distinct actions**, and every belief above **0.1** selects the same one (`latent/gpu/sleep`).
- This is **not** specific to the hand-written table. All **15 hardware-derived tables** (§P) saturate at
  the same point: the token term dominates the latency term, so one representation wins outright once an
  item is reused at all. Varying `_TIER_READ_MS` cannot fix it.
- Raising `_REPR_TOKENS[LATENT]` off its placeholder of 0 shifts the threshold but still leaves **at most
  3 actions**.
- On the real toolagent trace, 6 distinct actions are emitted — but the variety comes from **capacity
  binding** as tiers fill in sort order, not from the belief separating items. 31% of blocks carry a
  belief above the saturation point; for those the belief affects **only their queue position**.

**Consequence.** §5.2's rank-normalization control ("ordering is the operative mechanism") is largely a
*prediction of the setup*, not a discovery about the workload. What §5.2 measures is narrower than
"prediction does not help placement": it is *a better ordering does not buy much when the action attached
to each position is fixed and the only scarce resource is tier capacity*. The paper now says exactly
that, in §5.2, before the reader reaches the controls. `tests/test_action_degeneracy.py` pins both facts
so the limitation cannot silently expire.

This also answers Reviewer E's R1 (below): sweeping the cost model for the *captured* fraction would not
have resolved the concern, because every table in the sweep shares the degenerate action structure. The
free parameter that matters is `_REPR_TOKENS`, not `_TIER_READ_MS`.

### S.2 The abstract called an oracle-over-policies "purely reactive" (Reviewer E, R4)

Verified. `best_reactive_over_belady` is a per-operating-point argmax over **seven** policies including
LeCaR and LRB, and at a 2% tier on **both** traces the maximum is attained by **LeCaR**, a learner. The
abstract and intro described that band as "purely reactive tiering, forecasting nothing".

Fixed: the abstract and intro now report the band for rules that genuinely forecast nothing (LRU, LFU,
fill-once): **75–99% / 40–94%**, and state separately that admitting the adaptive and learned policies
widens the envelope only to 76–99% / 42–94%. `check_paper_numbers.py` now pins **both** bands, so the
conflation cannot recur.

### S.3 §4 forward-referenced a retracted result (Reviewer E, R2)

Verified: §4 closed by promising that §5 "shows the same features succeed or fail by a wide margin
depending on whether the serving interface exposes request timing" — the thesis retracted in §I on
2026-08-13. Replaced with a statement that none of the architecture is evaluated.

### S.4 Table 1 claimed six learned capabilities for an unevaluated system (Reviewer E, R2)

Verified: STOA carried `\checkmark` (defined in the caption as "learned") in all six columns, while the
GNN policy, factored heads, budget conditioning and joint credit are evaluated **only on 64-item
synthetic Zipf traces** whose artifacts the paper never cites. Fixed: the row is now marked `∗` =
*proposed, not evaluated in this paper*, with the caption saying so and pointing at §5 for the reasons to
doubt parts of it.

### S.5 C5 overclaimed the representation axis (Reviewer E, R3)

Verified: `MODES = (plaintext, summary, extract)` — all three are **textual**. `VECTOR`, `GRAPH`,
`LATENT` and `PARAMETER` are never instantiated anywhere in this work. C5 said "the first exercise of the
representation axis"; it is now "a first probe ... not the vector, graph, latent or parameter forms,
which we never instantiate". §5.7 also called a trade-off "real" two paragraphs before reporting 0 of 8
comparisons significant; that now reads "visible ... clears no significance threshold".

### S.6 Reviewer E's standing objections that we have NOT resolved

- **R1 (cost model never swept for `captured`).** The sweep now computes it
  (`run_calibration_sensitivity.py` gained a learned arm), but see §S.1: all 15 tables share the
  degenerate action structure, so the sweep bounds less than it appears to. The honest resolution is
  §S.1's re-scoping, not the sweep.
- **R5 (one vendor).** Unresolved and probably unresolvable with available public traces.
- **R7 (defect discovery had not decayed at submission).** This round found five more. That is evidence
  for the reviewer's point, not against it.

Reviewer E's score was **reject, borderline**, with the fixes above listed as what would move it. Four of
the six specific complaints are now fixed; the two that remain are honest limitations rather than errors.

## T. C2 IS A NORMALIZATION ARTIFACT — THE DECOUPLING CLAIM DOES NOT SURVIVE (2026-08-16)

Reviewer C's finding, **independently reproduced here**. This retracts the paper's central contribution.

### T.1 The mechanism

`scripts/train_session_features.py:77` skips any block with no pre-split access
(`past = [s for s in steps if s < split]; if not past: continue`). So a model's belief covers only
blocks that already exist at the split instant. `place_with_beliefs` assigns every other block
`believed.get(id, 0.0)`. But the oracle is scored over **all** `wl.items` with true future counts.

Oracle and policy are therefore scored on different item sets, and the difference is not small:

| | conversation | toolagent |
|---|---|---|
| blocks with no pre-split access | 82,659 / 182,790 (45.2%) | 84,465 / 183,300 (46.1%) |
| future accesses they carry | 111,361 | 113,147 |
| full oracle gap `c_pref − c_orac` | 160,199.8 | 162,601.8 |
| **attainable gap** (clairvoyant belief on scored blocks only) | **4,995.8** | **3,900.3** |
| **share of the gap unreachable by construction** | **96.9%** | **97.6%** |

### T.2 What the corrected normalization says

The learned policy captures +2.31% / +2.10% of the *full* gap. As a fraction of what is *attainable*:

| trace | reqs | AUC | captured (full gap) | attainable ceiling | **% of ceiling** |
|---|---|---|---|---|---|
| conversation | 1,500 | 0.621 | 0.76% | 4.74% | **16%** |
| toolagent | 1,500 | 0.566 | 1.32% | 5.00% | **27%** |
| conversation | 6,000 | 0.769 | 2.78% | 4.62% | **60%** |
| toolagent | 6,000 | 0.765 | 1.52% | 4.12% | **37%** |
| conversation | 12,031 | 0.812 | 2.31% | 3.12% | **74%** |
| toolagent | 23,608 | 0.925 | 2.10% | 2.40% | **88%** |

AUC 0.57 → 0.93 maps to **16% → 88% of attainable benefit, monotonically**, while the ceiling itself
falls. **That is coupling, not decoupling.** The reported interval `[+0.8%, +2.9%]` looks flat only
because its denominator is a fixed ~97% that no causal policy can reach: the oracle's advantage on blocks
that do not exist yet when the decision is made.

**C2 as written is false, and the title "Ranking Is Not Placement" is falsified by the corrected
normalization.**

### T.3 Three further defects in the same metric, each larger than the reported effect

- **Tie-break by enumeration order (`sequential.py:222-228`).** At `n_hat = 0` every action costs 0 and
  `if c < best_c` keeps the first, which is DISK. Adding `1e-9` to every belief — an operation that
  cannot change any ordering or any real decision — **changes the baseline by 35.6%** (verified:
  12711.3 → 8191.5). Every zero-belief block is parked on the slowest tier while remote sits empty.
- **Magnitude, not just order, selects the action.** Thresholds at n̂ ≈ 7.4e-4, 0.079, 0.41. The ladder's
  `minmax × mean_reuse` transform moves items across them differently per model, so the rungs are not on
  a common scale. The `log(prefix count) alone` rung has the *same ordering* as the reference it is
  scored against and still reports −3.5%, which is a direct proof the metric is contaminated.
- **Capacity is not scarce for the ranked set.** Reused scored blocks: 7,248 / 4,901 against 73,116 /
  73,320 GPU+CPU slots — 10–15× oversupplied. The perfect-count and perfect-binary oracles give
  *identical* cost. So §S.1's fallback framing ("the value of ordering under scarce capacity") also
  fails: capacity never binds on the positives.

### T.4 Two controls cited in §5.2 were never run

- *"Rank-normalizing every belief ... changes the captured headroom by about two points."* **No script in
  the repository implements rank normalization** (verified: no match for rank-norm/rankdata/double-argsort
  anywhere in `src/` or `scripts/`). A magnitude-controlled re-run gives up to **28 points** on a belief
  that has any effect at all.
- *"Substituting true future counts as the belief recovers 95\% of the interval."* No artifact contains
  this figure. `reference_costs` computes the oracle via `_oracle_cost`, a different code path from
  `place_with_beliefs`, so the two numbers are not even the same construction.

`check_paper_numbers.py` passed 52/52 while both claims were uncheckable — its coverage is exactly
complementary to the claims that carry the argument.

### T.5 Not defects (verified, worth stating)

- All items have identical `size_bytes` (8192), so `place_with_beliefs` ignoring size is harmless *here*.
  It becomes wrong the moment sizes vary; the paper should say so rather than leave the greedy looking
  general.
- The oracle is exactly optimal under equal sizes (exchange argument on the per-tier slot assignment).
  The 81–87% gap is measured against a genuine optimum — it is simply the wrong denominator for a
  causal policy.

## U. THE LRB COLUMN MEASURED RANDOM EVICTION (2026-08-16)

Reviewer B's finding, **independently reproduced here by instrumenting `GBDT.fit`**.

### U.1 The bug

`src/stoa/lrb.py` truncated the training buffer to its **tail**:

```python
if len(train_X) > max_train_rows:      # 60,000
    train_X = train_X[-max_train_rows:]
```

Labelled rows arrive one at a time as blocks are re-accessed. Censored rows arrive in **one contiguous
burst of ~600k per checkpoint**. So the tail held only the newest censored burst. Instrumented:

```
fit 1: distinct_y = 14018   y_std = 1.0115    <- real labels
fit 2: distinct_y = 24787   y_std = 1.1319    <- real labels
fit 3: distinct_y =     1   y_std = 0.0000    <- collapsed
...  11 of 13 fits identical
```

A constant target makes the regressor return a constant, so
`max(range(k), key=lambda j: scores[j])` returns `j = 0` and the victim is `idxs[0]` from
`rng.sample(...)` — **uniform random eviction, for roughly the last 70% of every trace.**

Reviewer B's random-replacement control (5 seeds) reproduces the published LRB column to within 1.5
points at all eight operating points and *beats* it at five. `experiments/lrb_sweep.json` inherits the
same bug, so §5.3's twelve-point hyperparameter grid swept a random policy six ways.

The published diagnostic is also wrong in kind: §5.3 reports "77–83% of LRB's training rows are
censored", a run-wide aggregate. The rows actually fitted on were **100%** censored at 11 of 13 fits.

### U.2 The fix, and why the column is still not citable

Applied: uniform subsample of the buffer through a **dedicated RNG** (drawing from the policy's own
`rng` would consume eviction-sampling draws and change the policy while claiming to fix its training),
plus a hard failure when label variance is zero. Labels recover at every fit
(`distinct_y` 3,004–26,232, `y_std` 0.54–1.37).

| trace | tier | published (broken) | refit | delta | ARC |
|---|---|---|---|---|---|
| conversation | 2% | 27.9% | 27.6% | −0.3 | 29.4% |
| | 5% | 44.3% | **32.2%** | **−12.1** | 57.9% |
| | 10% | 63.9% | **45.0%** | **−18.9** | 77.5% |
| | 20% | 78.2% | 75.8% | −2.4 | 87.4% |
| toolagent | 2% | 70.2% | 69.0% | −1.2 | 70.7% |
| | 5% | 76.5% | 71.1% | −5.4 | 83.6% |
| | 10% | 84.7% | 80.9% | −3.8 | 92.3% |
| | 20% | 91.7% | **94.9%** | **+3.2** | 96.2% |

**But the fix is a research decision, not a correction.** Reviewer B fixed the same bug by reservoir
sampling over the *memory window*; at conversation/10% that gives ≈63%, where sampling over *all
accumulated rows* gives 45% — a **19-point disagreement between two defensible readings of the published
design**. Ours goes stale (old rows accumulate); theirs discards history LRB's window is meant to retain.

**The LRB column is therefore WITHHELD** (`experiments/lrb_retraction.json`) until the sampling
semantics are settled against the NSDI'20 text. The paper's conclusion — LRB below ARC at every
operating point — survives under both fixes, but its published *numbers* do not, and §5.3's causal story
(censoring) is wrong: censoring stays at ~81% in the fixed run, so the truncation bug, not censoring,
cost the points.

### U.3 The baseline is a hindsight oracle too

"Best plain rule" is `max(LRU, LFU, fill-once)` chosen **per operating point with hindsight** — the same
argmax construction the paper already disclaims for the envelope bar, and already fixed once on the
numerator side (§S.2). Against a deployable fixed rule the margin is far larger:

| baseline | max margin of the best adaptive policy, 8 points |
|---|---|
| best-pure (per-point oracle) | +4.6 |
| **always-LRU (deployable)** | **+17.3** |
| **always-LFU (deployable)** | **+12.1** |

Recency-vs-frequency *inverts between the two traces* — the paper says so itself — and removing that
choice is exactly what ARC and LeCaR are for. C3's headline sentence is arithmetically true and
rhetorically inverted. Both baselines must be reported.

### U.4 What Reviewer B verified as CORRECT

- **ARC is faithful.** Clean-room transcription of Megiddo & Modha Fig. 4, structural invariants asserted
  over 698k accesses, agreement to 4 decimal places at all 8 operating points. The one deviation
  (integer floor δ) *helps* ARC by ≤0.27 points. The paper should say this more loudly than it does.
- **`simulate_fast` ≡ `simulate`** on real-trace slices (85–95% singletons — the regime the existing test
  does not cover), and `simulate_fast("belady")` matches brute-force OPT on 120/120 cases.
- LeCaR/CACHEUS deviations (shared ghost list, penalize-vs-reward update, CACHEUS initial learning rate)
  were each measured and are immaterial: ≤0.1, algebraically identical, and ≤2.6 points respectively.

### U.5 Two further defects

- **`simulate_fast(wl, "learned", c)` silently returns LFU** labelled `policy='learned'`: `"learned"` is
  in `POLICIES` so it passes validation, then falls through `key_of` to the LFU branch. No paper number
  uses it today. It should raise.
- **Six of the eight operating points have zero Belady capacity misses.** `cache_frac` is a fraction of
  the whole item universe, of which only 21–24% is ever reused, so at 5/10/20% the tier holds 21–93% of
  the reusable working set and Belady's hit rate is pinned at the compulsory-miss ceiling. Extending to
  0.2/0.5/1% — where the constraint actually binds — **strengthens the paper**: the adaptive family loses
  to the best pure rule by up to **7.25 points** there. Those rows are the paper's best evidence and are
  missing.

## V. THE METRIC, REBUILT — AND WHAT SURVIVES (2026-08-17)

All four metric defects from §T/§U are fixed in `src/stoa/sequential.py`, pinned by
`tests/test_metric_fixes.py` (7 tests):

1. **Tie-break by tier speed, not enum order.** Adding 1e-9 to every belief moved the baseline 35.6%;
   it now moves it 0.00%.
2. **`attainable_ceiling(wl, scored_ids)`** — the clairvoyant belief restricted to blocks a causal
   policy could have an opinion about. `captured_fractions()` returns both normalizations so neither can
   be quoted without the other.
3. **`quantile_match(scores, reference)`** — puts every policy's belief on one magnitude multiset so
   only the ordering varies. **With mid-rank tie handling**: without it a constant score vector became
   strictly increasing in input order, and the zero-information control acquired a systematic ranking and
   beat a real model. Same defect class as the AUC tie bug of §E.
4. Deployable baselines (always-LRU / always-LFU) recorded alongside the per-point oracle (§U.3).

### V.1 The capacity ladder under the fixed metric: monotone in AUC, and all of it negative

> **SUPERSEDED by §AC (2026-08-26).** The artifact behind this table predated the tie-break fix of §W
> reaching all three placement routines. `constant` is −46.1 / −60.6, not −154.5 / −825.7, and
> monotonicity in AUC holds on toolagent only. The numbers below are left as recorded; do not cite them.

`experiments/capacity_ladder_fixed.json`, full traces:

| rung | AUC | % of attainable (conversation) | % of attainable (toolagent) |
|---|---|---|---|
| constant | 0.500 | −154.5% | −825.7% |
| random | 0.49 | −208.2% | −838.7% |
| log(count) | 0.64 / 0.71 | −70.9% | −75.8% |
| linear | 0.81 / 0.93 | −18.8% | −32.1% |
| GBDT | 0.80 / 0.91 | −23.9% | −33.0% |
| MLP | 0.82 / 0.93 | −8.5% | −19.3% |
| ceiling | 1.000 | 100% | 100% |

**Reviewer C's coupling claim is confirmed in direction**: the damage falls monotonically as AUC rises
(−154 → −71 → −19 → −8.5). But the level is not what they computed: every arm is *below* the
prefix-greedy baseline. The difference is that their numbers used the unfixed metric (min-max × mean
reuse, old tie-break); ours quantile-match to the oracle's magnitude multiset.

**This is itself a finding, and it is the reviewers' point made sharply**: the same experiment yields
"learned models capture 74–88%" or "every model is 8–33% worse than counting" depending on a choice —
which magnitude scale to place beliefs on — that no version of this paper ever stated. A quantity that
flips sign under an unstated convention is not yet a measurement. We report both and claim neither.

### V.2 The arrival experiment: the headroom is real, and prediction is not what captures it

> **SUPERSEDED by §AC (2026-08-26).** Same cause: the artifact predated the §W fix to the routines that
> produce the *denominator*. Current values are reachable share **83.5% / 19.0%** (27× and 8×, not 29×
> and 22×), FCFS at **87.4% / 50.2%** of the ceiling, and shuffling the arrival order costs **3.3 /
> 14.1** points — not 1.8 / 3.0. The second bullet below ("FCFS ordering contributes little") is
> **falsified on toolagent**: arrival order is worth 14 points there. Left as recorded; do not cite.

`scripts/run_arrival_admission.py`, `experiments/arrival_admission.json`, full traces. Blocks are placed
at **first appearance** from request-level context (session age/rate/recency, position in request,
request length, warm-up), fitted on the first half of arrivals and placed on the second. No feature reads
the future.

| trace | arm | AUC | % of full gap | % of attainable |
|---|---|---|---|---|
| conversation | arrival-learned | 0.574 | 82.8% | 91.6% |
| | **arrival-FCFS** | 0.500 | **84.2%** | **93.2%** |
| | arrival-shuffled | 0.500 | 82.6% | 91.4% |
| | ceiling | 1.000 | 90.4% | 100% |
| toolagent | arrival-learned | 0.635 | 47.0% | 88.9% |
| | **arrival-FCFS** | 0.500 | **47.3%** | **89.5%** |
| | arrival-shuffled | 0.500 | 45.8% | 86.6% |
| | ceiling | 1.000 | 52.8% | 100% |

Three things, and they change the paper:

- **Moving the decision to arrival makes the headroom reachable.** The split-instant protocol could
  attain 3.1% / 2.4% of the hindsight gap. Deciding on arrival attains **90.4% / 52.8%** — 29× and 22×
  more. The benefit was never absent; the protocol could not reach it.
- **Deciding at all captures ~90% of it. Predicting does not add.** FCFS admission with a fixed action
  reaches 93.2% / 89.5% of the ceiling; the learned arm (AUC 0.574 / 0.635) reaches 91.6% / 88.9% —
  *slightly worse*. Shuffling the arrival order costs 1.8 / 3.0 points, so FCFS ordering contributes
  little; almost all of the benefit is in **when** the decision is made, not in what it decides.
- **"Ranking is not placement" survives, in a corrected and sharper form.** The original claim was
  measured where 97% of the benefit was unreachable. Restated: *at the point where the headroom actually
  lives, timing of the decision captures ~90% of it and prediction quality adds nothing on top.* That has
  a design implication the original did not: build an admission point, not a predictor.

### V.3 What this costs the paper

C2 as written is retracted (§T). The replacement is stronger but different, and the title must change:
the result is now about **when** placement is decided, not about whether ranking converts. §5.2's
capacity ladder becomes a secondary result reported under two normalizations with the convention stated.
The §5.2 controls that were never run (§T.4) must be run or deleted.

## W. THE TIE RULE WAS IN THREE ROUTINES, AND I FIXED ONE (2026-08-17)

While rewriting on the corrected metric, two of my own scripts disagreed about the same denominator:
`run_belief_controls.py` put the attainable ceiling at 44.8% of the full gap, `run_capacity_ladder.py`
at 5.3%. The cause was that the tie-break fix of §V had been applied to `place_with_beliefs` only.

Three routines choose among equal-cost actions, and a block with no future accesses makes every action
cost the same, so the choice is **pure tie-breaking**:

| routine | used as | had the defect |
|---|---|---|
| `place_with_beliefs` | the policy under test (numerator) | fixed in §V |
| `prefix_greedy_cost` | the no-learning baseline (denominator) | **still broken** |
| `_oracle_cost` | the hindsight reference (denominator) | **still broken** |

So after §V the numerator and denominator of every reported fraction were computed under **different
rules**. Both are now fixed, and `tests/test_metric_fixes.py` pins the agreement two ways: the two
baseline routines must return the same cost, and a clairvoyant belief through `place_with_beliefs` must
equal the oracle `reference_costs` computes by the other path.

The lesson is narrower and more useful than "fix your tie-breaks": **when a defect is in a shared
convention rather than in a function, fixing one call site makes the system less consistent than leaving
it alone.** Nothing failed. Both scripts ran, both produced plausible numbers, and the only symptom was
two artifacts quietly disagreeing about a quantity neither of them reported directly.

## X. LRB SAMPLING SEMANTICS SETTLED (2026-08-17)

§U left the LRB column withheld because two defensible repairs of the truncation bug disagreed by 19
points: subsampling **all accumulated rows** versus subsampling the **memory window**. Resolved in favour
of the window, which is what the published design's "sliding memory window" means — rows older than
`memory_window` requests are now dropped by age first, and the uniform subsample only caps memory once
the window itself exceeds the budget. Sampling all history is not a sliding window; it accumulates stale
rows without bound, which is a different algorithm.

## Y. REPRODUCIBILITY PACKAGING (2026-08-17)

Reviewer D's blockers, closed:

- **`requirements-lock.txt`** records the exact reference environment (torch 2.13.0+cpu, numpy 2.2.6,
  httpx 0.28.1, pytest 9.1.1, matplotlib 3.10.9). `pyproject.toml` keeps loose bounds so the package
  stays installable; the lockfile is the provenance record.
- **`REPRODUCIBILITY.md` §3b** now states that scipy/sklearn/LightGBM are absent **by design with no
  fallback path**, so a reproducer who has them installed gets identical numbers — and that the GBDT rung
  is consequently not `HistGradientBoostingRegressor` (200 depth-4 trees, 64 bins, no early stopping vs
  sklearn's leaf-wise 31 leaves with an auto-enabled holdout above 10k rows; the ladder trains on
  30k–55k rows).
- **API cost is stated**: ~7,200 chat completions for the LoCoMo run and ~2,400 for the representation
  axis, roughly $3–5 and $1 at gpt-4o-mini list pricing.
- **`torch.set_num_threads(1)`** in the capacity ladder: the MLP rung's cross-trace AUC moved by 0.006
  between 1 and 112 threads, and that rung carries a third of one claim.
- **11 superseded artifacts quarantined** to `experiments/superseded/` with a README naming what each
  was and which retraction it belongs to. Deleting them would erase the record; leaving them in place let
  a reader mistake them for current. Nothing in the paper, the checker or REPRODUCIBILITY.md reads that
  directory, and no current artifact is now unreferenced.
- The checker **fails loudly on an empty `experiments/`** rather than reporting a vacuous pass, and
  refuses to let the three retracted claims reappear in the paper without a retraction note.

## Z. THE SLIDING WINDOW WAS A NO-OP, AND THE ONLY SYMPTOM WAS AGREEMENT (2026-08-17)

§X settled the LRB sampling semantics in favour of a sliding memory window and implemented one. The
re-run produced numbers **byte-identical to the unfiltered version** — 27.6 / 32.2 / 45.0% of Belady at
2/5/10% on conversation. That identity was the only evidence anything was wrong.

The guard read `if train_t and train_t[0] < cutoff_t`, which is the oldest row **only while the buffer is
sorted**. The uniform subsample that runs immediately below it reorders the buffer, so from the first
time it fired, `train_t[0]` was an arbitrary element and the age filter stopped firing. Fixed to
`min(train_t)`, and `tests/test_lrb.py::test_the_training_buffer_is_actually_bounded_in_age` now asserts
the buffer stops growing across retrains under a window far smaller than the trace.

Two things worth keeping from this. First, **a fix that changes nothing is a finding, not a relief** —
the natural reading of an unchanged number is "the window does not bind here", and that reading was
wrong. Second, this is the same shape as §W: a correct local change (add an age filter) interacting with
an existing one (subsample reorders) to produce a silent no-op. Neither is visible in any aggregate; both
were caught only by comparing a number against what it should have been able to be.

Running total of defects found in this codebase whose only symptom was a plausible number: **fourteen.**

## AA. THE CHECKER'S OWN COVERAGE GUARD WAS UNDERCOUNTING (2026-08-19)

The guard added in §Y --- "fail if fewer than N claims were checked, because a pass with too few checks
means artifacts are missing" --- sat in the middle of the file. Every claim block added after it was
therefore invisible to it: `checked` was 35 at the guard's line while the final report printed 66.

The failure mode is the guard's own: it passed while a third of the paper went unchecked, and the only
symptom was a mismatch between two numbers it printed itself. Moved to the end of the file, where
`checked` is final, and the floor raised from 36 to 60.

This is the fifteenth defect in this codebase whose only symptom was a plausible number, and the second
in a tool built specifically to catch that class.

## AB. VENUE AND SUBMISSION TIMING (2026-08-19)

**PVLDB Volume 20, EA&B track.** Verified from vldb.org/2027/submission-guidelines.html: monthly rolling
deadline, 1st of each month 5 PM PT, mandatory abstract by the 25th of the prior month, running through
**2027-03-01**. 12 pages excluding references. EA&B requires the reproducibility package link **in the
initial submission** ("there are no excuses").

The fit improved with the rewrite. The earlier draft was an unevaluated system plus negative results; the
current one is coherently a measurement paper, with §§3--4 explicitly marked as proposed-not-evaluated.

**Not the next cycle.** The central claim changed on 2026-08-16, the metric it rests on was rebuilt on
08-17, and rebuilding it surfaced three further defects (§W, §Z, §AA). Submitting on 09-01 would repeat
the failure this paper is about. The decision rule instead of a date: **run one complete re-verification
pass; if it surfaces nothing new, submit to the following cycle; if it surfaces anything, repeat.**
Target 2026-10-01 (abstract 09-25), fall back to 11-01. PVLDB's rolling structure means deferring a cycle
costs nothing, which is why it is the right venue for work in this state.

Blocking on the author: `\vldbavailabilityurl` still reads ANONYMIZED, and EA&B will not accept a
submission without it.

### AB.1 Reframing done in this pass

The methodology contribution read as an inventory of our own mistakes, which Reviewer E identified as a
weak contribution and a corrosive one to reward. Repositioned as a claim about how this decision is
studied generally: **split-instant evaluation understates placement on any workload that admits new items
after the split, and the diagnostic is one extra placement pass** (`attainable_ceiling`). Our retractions
now appear as evidence for how invisible the failure is, not as the contribution itself. The subtitle
changed from "the protocol that took three retractions to reach" to a statement of the finding.

## AC. THE RE-VERIFICATION PASS FOUND THE PAPER CARRYING PRE-FIX NUMBERS (2026-08-29)

§AB set a decision rule: run one complete re-verification pass; submit to the next cycle only if it
surfaces nothing. It surfaced something on the first check, and the finding is uncomfortable in a
specific way — the *code* was correct and the *paper* was not.

§W fixed the tie-break in all three placement routines on 08-18. Three artifacts the paper cites were
generated **before** that: `arrival_admission.json` and `capacity_ladder_fixed.json` (08-17) and
`calibration_sensitivity.json` (08-14). Two of those routines are **denominators**. Regenerating them
moved the headline table substantially.

### AC.1 What moved

| quantity | paper carried | regenerated |
|---|---|---|
| conversation reachable share | 90.4% | **83.5%** |
| toolagent reachable share | 52.8% | **19.0%** |
| conversation FCFS (of attainable) | 93.2% | **87.4%** |
| toolagent FCFS (of attainable) | 89.5% | **50.2%** |
| ladder `constant` (conv / tool) | −154.6 / −825.7 | **−46.1 / −60.6** |
| ladder `log(count)` (tool) | −75.8 | **−59.1** |

### AC.2 Two claims falsified, two survive

- **Survives** — *timing recovers the benefit*: 3.1% → 83.5% and 2.4% → 19.0%, i.e. 27× and 8×. The
  toolagent factor was reported as 22×; it is 8×.
- **Survives** — *learning on top adds nothing*: FCFS beats the learned arm on both traces
  (87.4 vs 84.5, 50.2 vs 47.2). This is the only claim that has survived every correction to the metric.
- **FALSIFIED** — *"FCFS captures ~90% of what is reachable"*. On toolagent it captures **50.2%**.
- **FALSIFIED** — *"shuffling the arrival order costs under three points"*. It costs 3.3 on conversation
  and **14.1 on toolagent**. First-come-first-served is doing real work there, and the paper explicitly
  said it was not. The design implication narrows from "deciding at all is enough" to "decide early, and
  do not assume the order is incidental."
- Also narrowed: the ladder's monotonicity now holds on **toolagent only**; on conversation the constant
  belief (−46.1) beats log(count) (−70.7) despite carrying no information.

### AC.3 What the checker did and did not do

The checker **caught it**: three ladder rows reported MISS and the monotonicity assertion fired, exit 1.
That is the tool working.

But it caught it only because the artifacts were regenerated. Nothing in the pipeline noticed that an
artifact was **older than the code that produces it** — the checker compares artifacts to the paper, not
artifacts to the code that made them. A staleness guard belongs in it: refuse to pass when any artifact
predates the module it depends on. Without that, "the code is fixed" and "the paper is fixed" remain two
different states, and this project has now been in the gap between them twice (§W, and here).

### AC.4 The rule applies to itself

By §AB's rule the pass surfaced something, so the paper does not go to the next cycle. `calibration_
sensitivity.json` is still regenerating; when it lands, the pass restarts from the top. Sixteen defects,
and the seventeenth was the belief that fixing code fixes results.

## §AD — figures are artifacts too, and nothing checked them

The re-verification pass of §AC ended clean: 173 tests, 66/66 checks, every artifact newer than the
code that produced it. The paper built at 12 pages with no undefined references. That was wrong.

`paper/figs/fig_reactive.pdf` and `fig_capacity_ladder.pdf` were **fifteen days older** than the
artifacts they plot. The paper embeds the PDF, not the JSON. So §5.3's text carried the corrected
ladder while Figure 2 still drew the pre-fix one, and the checker — which compares artifacts to the
*paper source* — cannot see inside a PDF. Regenerating both changed both.

The staleness guard of §AC was built one level too high. It asked "is this artifact older than the
code that produced it?" and stopped. It did not ask the same question of the next link in the chain,
where the artifact is itself a source. The chain is:

    src/*.py  ->  experiments/*.json  ->  paper/figs/*.pdf  ->  main.pdf

§AC guarded the first arrow. `FIG_SOURCES` in `check_paper_numbers.py` now guards the second, and it
was verified by `touch`-ing an artifact and confirming the guard fires — not by observing that it
passed.

**The generalization**: a freshness check on one edge of a derivation chain reads as a freshness check
on the chain. It is not. Every arrow needs its own, and a guard that has only ever passed has not been
tested. This is the eighteenth defect whose only symptom was a plausible-looking output.

## §AE — the checker verified a table and not the sentence about it

The pass of §AD ended clean and was committed. Re-running it a week later, the standard three steps
(173 tests, 66/66 checks, a 12-page build with no undefined references) passed again. Two things were
wrong anyway, and neither was in the checker's reach.

**1. The repository's front page carried the falsified numbers.** `README.md` — the first file anyone
opens on GitHub — still said the reachable share rises to *90%* and *53%* and that FCFS captures *93%*
and *90%*. Those are the pre-§AC values; the true ones are 83.5 / 19.0 and 87.4 / 50.2. §AC corrected
`paper/README.md` and `REPRODUCIBILITY.md` and never touched the root README, because the sweep that
found the stale prose was written by listing the files I remembered editing.

**2. §5.8 stated a number no check covered.** It said *"two independent repairs of that bug disagree by
19 points."* The artifact says the two repairs disagree by at most **2.1** points; the **17**-point
figure is the gap to an independent reimplementation, which is a different quantity. §5.5 states both
correctly. §5.8 compressed them into one number that is neither.

The checker verified all eight rows of the LRB band table and passed. It had no check on a *sentence
about* those rows. The row-level checks cannot catch this: every individual number in the table was
right, and the summary claim was still false.

I preserved that sentence verbatim while editing the paragraph around it in §AD, which is the specific
failure worth naming — **an edit that leaves a number untouched is not a verification of it.** The
number arrived in the draft before the checker had an assertion for it, and editing near it created a
false impression of having reviewed it.

Both are now checked: `widest disagreement between the two repairs` (2.1) and `independent
reimplementation gap` (17) are derived from `lrb_retraction.json` and asserted against the paper text.
The first version of the second check matched the `/10%` inside "conversation/10%" and reported a
53-point gap — a reminder that a check written carelessly fails loudly, which is the good case.

**The generalization**: a checker that verifies every element of a table certifies the table, not the
prose that summarizes it. Aggregate statements — widths, spans, maxima, "at most", "by a factor of" ---
need their own assertions, derived from the same artifact. This is the nineteenth defect whose only
symptom was a plausible number.

## §AF — the notebook stated falsified claims in the present tense

Pass 4. The standard three steps passed again (173 tests, 68/68 checks, 12 pages, no undefined
references). This time the coverage sweep was written **mechanically** rather than from memory — the
specific failure of §AE — by enumerating every tracked file containing any of the paper's headline
quantities and classifying each as checked, guarded, or neither. Three files came back unguarded:
`docs/claims_dependency.md`, `docs/research_plan.md`, and the preserved superseded draft
`paper/main_decoupling_retracted.tex`. The last two are legitimately historical. The notebook was not.

§V.1 and §V.2 — this document — stated the pre-§AC values **in the present tense with no forward
pointer**: "Deciding on arrival attains 90.4% / 52.8% — 29x and 22x more", and

> "Shuffling the arrival order costs 1.8 / 3.0 points, so FCFS ordering contributes little; almost all
> of the benefit is in **when** the decision is made, not in what it decides."

That is precisely the claim §AC falsified: 14.1 points on toolagent. A reader landing on §V.2 — 150
lines before the correction — reads a falsified conclusion as a finding.

A lab notebook must record superseded measurements; editing them out destroys the record's value, which
is the whole reason this file exists. So the fix is a banner, not a rewrite: §V.1 and §V.2 now carry
**SUPERSEDED by §AC** headers giving the current values, with the original text left intact beneath.

`check_paper_numbers.py` now enforces it. Any occurrence of a known-superseded headline value in this
file must appear in a section carrying a SUPERSEDED / RETRACTED / FALSIFIED / "do not cite" marker, or
under an explicit "paper carried | regenerated" before-and-after column. Verified by deleting the §V.2
banner and confirming eight findings, after a first attempt at that verification failed to remove the
whole banner and wrongly appeared to show the guard was dead — **a guard test that leaves part of the
guard's input in place tests nothing**, the same shape as §AE's "an edit that leaves a number untouched
is not a verification of it."

**The generalization**: correcting a claim in the document that makes it does not correct the documents
that recorded it. Each surface needs either an update or a marker, and which one depends on whether the
surface is an assertion or a record. This is the twentieth defect whose only symptom was a plausible
number.
