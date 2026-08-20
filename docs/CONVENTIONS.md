# Codebase conventions

How this repository is organized and the rules it holds itself to. Written for anyone extending or
reproducing the work; `README.md` is the entry point, `REPRODUCIBILITY.md` maps claims to commands, and
`claims_dependency.md` in this directory is the lab notebook.

Several conventions below look fussier than they need to be. Each exists because breaking it produced a
plausible-looking wrong number at some point in this project's history; `claims_dependency.md` records
which.

## What this is

STOA (**S**torage & **T**iered **O**rchestration for **A**gents) is a **research project**, not a
shipping product. It proposes a *learned memory-tier orchestrator* for LLM agents that jointly
decides, per memory item, its **representation × storage tier × update timing** under
latency/cost/token budgets, framed as a single budget-constrained MDP.

The current `src/stoa` tree is an **interface scaffold (stubs)**. Modules define the shapes from the
design doc; most behavior is placeholder logic marked with `TODO(Mx-y)` pointing at roadmap
milestones. When implementing, fill stubs in roadmap order (M0→M18), not opportunistically.

**The design docs are the source of truth**, and code comments cite them by section
(e.g. `docs/stoa_design.md §2`). Before changing a module's semantics, read the referenced section.
- `docs/stoa_design.md` — system design + MDP formalization (the spec the code implements)
- `docs/research_plan.md` — focused related-work survey + operational research plan (experiment
  matrix, milestone go/no-go gates, code↔plan mapping); the detailed expansion of the design doc
- `docs/survey.md` — 114-paper literature basis; `docs/references.bib` — verified BibTeX
- `ROADMAP.md` — milestones M0–M18, each with deliverables and research questions (RQ1–3)

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # editable install + pytest
pip install -e ".[rl]"           # + torch (CPU) & numpy — needed for the GNN policy (M6-9)

pytest -q                        # run all tests (GNN tests skip cleanly without torch)
pytest tests/test_environment.py::test_action_space_is_masked   # single test

# smoke-check the scaffold is importable / wired (no third-party deps needed)
python -c "from stoa import Orchestrator, ItemState, Tier; print(Orchestrator().step([ItemState('m0', tier=Tier.CPU)]))"

# reproduce the milestone deliverables (write JSON to experiments/)
python scripts/build_dataset.py        # M0-3  oracle upper-bound baseline
python scripts/run_ablation.py         # M3-6  per-axis ablation (online tier + static repr)
python scripts/train_controller.py     # M6-9  linear Belady-imitation controller (RQ2)
python scripts/train_gnn_policy.py      # M6-9  GNN factored policy warm-start (RQ2, torch CPU)
python scripts/finetune_rl.py          # M6-9  REINFORCE fine-tune on top of BC (torch CPU)
python scripts/run_budget_frontier.py  # M9-12 budget-conditioned frontier (RQ3)

# production-trace experiments (need data/mooncake_*.jsonl; see src/stoa/mooncake.py)
python scripts/run_reactive_real.py       # reactive tiering incl. ARC, full traces
python scripts/run_capacity_ladder.py     # linear -> GBDT -> MLP -> oracle; the "not the model" control
python scripts/run_sampling_study.py      # randomized-draw control (CI on the kvct claim)
python scripts/run_reactive_real.py --lrb # + the LRB-style learned policy (~35 min)
python scripts/run_lrb_sweep.py           # LRB hyperparameter grid; the negative result is not tuning
python scripts/run_split_sensitivity.py   # sweeps the last free parameter (where the split falls)
python scripts/run_calibration_sensitivity.py  # tier RATIOS derived from hardware, then swept
python scripts/run_locomo_power.py        # what n the LoCoMo null would have needed (no LLM calls)
python scripts/check_paper_numbers.py     # asserts every paper number matches experiments/*.json

# LLM-backed task-utility (RQ1) — needs an OpenAI-compatible key in the ENV (never in code):
export OPENAI_API_KEY=sk-...            # environment only; never in a file inside this tree
pip install -e ".[eval]"
python scripts/run_memqa.py [--multihop]   # synthetic accuracy-vs-budget with a real LLM
python scripts/run_locomo.py               # real LoCoMo benchmark (auto-downloads data/)
python scripts/run_locomo_powered.py --judge      # all 10 dialogues, PAIRED tests (~2.5 h)
python scripts/run_representation_axis.py --judge # the representation axis, finally measured
```

Note: `pyproject.toml` sets `pythonpath = ["src"]` for pytest, so tests import `stoa` without an
install. Runtime `dependencies` are **deliberately empty**; extras add them per-milestone
(`[sim]` numpy, `[rl]` torch+numpy). **`import stoa` stays dependency-free** — torch lives only in
`gnn.py`/`train.py`, which are *not* imported by `stoa/__init__` (import them directly). The GNN
trains over the *simulator*, not an LLM, so **CPU is enough — no GPU** (torch-geometric is
intentionally unused; the small memory-graph GCN is hand-rolled in `gnn.py`).

## Architecture

The pipeline is a single-step control loop, runnable end-to-end even while backends are stubs:

`Orchestrator.step(items)` → `policy.act(items)` produces one `Action` per item →
`simulator.step(action)` returns realized `(latency, cost, tokens, freshness)` → `environment.reward(...)`
combines them via Lagrangian relaxation → returns aggregate metrics + `within_budget`.

Module roles (all under `src/stoa/`):
- **`environment.py`** — the MDP *interfaces*: the `Action` space (`Representation × Tier × Timing`
  enums), `ItemState`, `Budget` (L,C,B), and the `reward()` function. **Action legality lives here**
  in `Action.is_legal()` (e.g. PARAMETER promotion is SLEEP-only; LATENT only on GPU/CPU). This
  masking is load-bearing — `Action.space()` and the policy both depend on it, and tests assert it.
- **`simulator.py`** — MDP *transition dynamics*. Cost/latency/token tables (`_TIER_READ_MS`,
  `_REPR_TOKENS`, `_PROMOTE_COST`) are **illustrative placeholders** to be replaced with values fit
  from real LMCache-style traces in M0–3. Encodes the timing economics (sleep = amortized offline
  cost + off the serving path; noop = free).
- **`policy.py`** — the original no-op placeholder policy (keeps current tier), still used by
  `orchestrator.py`. The **real** factored GNN policy lives in **`gnn.py`** (kept out of the
  `__init__` import chain so `import stoa` needs no torch).
- **`orchestrator.py`** — the control plane; the `step()` unit the RL training loop will optimize.
- **`credit.py`** — credit assignment: Belady-oracle labelers — `belady_optimal_tier` (tier) and
  `belady_optimal_action` (full masked action), `weighted_cost` scalarizer — plus process-reward stub.

Implemented since the scaffold (fill order M0→M12; all CPU, no GPU):
- **`traces.py`** — synthetic workload/trace generator (Zipf popularity + optional temporal
  locality); stands in for real LMCache traces until M12-15.
- **`gnn.py`** (torch) — GCN trunk + factored `π_rep·π_tier·π_time` heads whose logits are summed
  over the 50 legal actions and softmaxed there (masking-exact) + a value head for RL.
- **`train.py`** (torch) — Belady behavioral-cloning warm-start for `gnn.py` (RQ2): co-access graph,
  node features, greedy-Belady labels, BC loop, held-out eval.
- **`rl.py`** (torch) — offline REINFORCE fine-tune of the BC policy with per-item credit + the value
  head as baseline. Honest finding: doesn't beat BC on the *static* placement (needs a sequential,
  occupancy-aware MDP); the machinery + diagnosis is the deliverable.
- **`learn.py`** — torch-free linear Belady-imitation reuse predictor (the online-eviction controller).
- **`lrb.py`** — LRB-style learned eviction (Song et al., NSDI'20) over `gbdt.py`: predicts
  time-to-next-request from inter-access deltas + decayed counters, samples candidates, retrains online.
  **Features are recorded at eviction-sampling time, never at request time** — doing the latter gives
  every training row recency 0 while inference always sees recency > 0, and cost 15 points of hit rate
  before it was found. `tests/test_lrb.py` pins it.
- **`experts.py`** — LeCaR (HotStorage'18) and CACHEUS (FAST'21): regret-weighted mixtures of an LRU and
  an LFU expert. CACHEUS's SR-LRU needs **both** an adaptive target size for the scan region R and a
  second-chance demotion from S into R; without them S grows without bound and the policy degenerates to
  a fill-once cache, scoring *below* LRU. Victim selection uses lazily-invalidated heaps (`_VictimHeaps`)
  because `min()` over the resident set does not finish at production cache sizes.
- **`calibration.py`** — derives `_TIER_READ_MS` from a model's attention shape (exact) and a declared
  device-class bandwidth (an assumption, not a measurement). Exists because scaling all tiers by one
  factor preserves their **ratios**, which is what a policy actually responds to; only varying device
  class tests them. `import stoa` stays dependency-free — this is pure arithmetic.
- **`stats.py`** — exact two-sided Fisher, **exact McNemar** for paired designs, and simulated
  power/reversal probabilities (scipy is unavailable here). Used to report the LoCoMo null **with the
  sample size that would have resolved it**. Every arm answers the *same* questions, so the LoCoMo
  comparison is paired: use `mcnemar_exact`, not `fisher_exact`, or the shared question difficulty
  swamps the arm difference.
- **`eval/representation.py`** — the representation axis (plaintext / summary / extract) as a rewrite of
  a `MemQATask`: same fact ids, same questions, same gold answers, different text and token cost.
  Budgets must be a fraction of the **plaintext** total for every mode, or a compressed representation
  silently gets a larger effective budget and the comparison is meaningless.
- **`gbdt.py`** — hand-rolled histogram gradient-boosted regression trees (numpy only; sklearn/LightGBM
  are unavailable here). Exists solely to supply a genuinely higher-capacity rung to the capacity ladder,
  so "the model was too weak" can be excluded rather than assumed. Squared loss, deterministic. `GBDT.flatten()` packs the ensemble into rectangular arrays and descends
  all trees at once — 7.7x faster on the ~64-row batches LRB scores per eviction, and bit-exact.
- **`eval/oracle.py`** — static offline references: oracle upper bound, capacity-feasible
  `greedy_belady_actions` (the BC target), naive baseline.
- **`eval/online.py`** — online tier-axis eviction ablation. `POLICIES` holds the **victim-rule** policies
  (Belady/learned/LRU/LFU-aliased-`h2o`/static) that `simulate`/`simulate_fast` dispatch on; **ARC is not
  in that tuple** — it is a four-list structure with its own state and its own entry point
  (`simulate_arc`), listed in `ALL_POLICIES`. Adding `arc` to `POLICIES` breaks every victim-rule caller.
- **`eval/budget.py`** — budget-conditioned placement frontier (RQ3).
- **`llm.py`** (httpx) — minimal OpenAI-compatible chat client; reads `$OPENAI_API_KEY` from the
  env (**never hardcode/commit keys**), not imported by `__init__`.
- **`eval/memqa.py`** — LLM-backed memory-QA under a token budget: measures the reward's task-utility
  `U` (real accuracy vs assumed 1.0) via an injected `answer_fn`, so it's mock-testable offline.
  Single-hop and two-hop (`generate_multihop_task`); supports per-fact token costs for real data.
- **`eval/locomo.py`** — LoCoMo adapter (arXiv:2402.17753): real 300-turn dialogues → budget-QA tasks
  (turns=facts, evidence=needed). `benchmarks.load('locomo')` uses it; data auto-downloaded to `data/`.
- **`eval/benchmarks.py`** — benchmark registry (LongMemEval, LoCoMo, MemoryAgentBench, Big ANN);
  `load('locomo')` works; the others still raise `NotImplementedError` until their adapters land.

`configs/default.yaml` mirrors the dataclass fields (`Budget`, `RewardWeights`, `PolicyConfig`,
simulator/training/eval) — keep it in sync when you add or rename config fields.

## Conventions

- **Stubs stay honest**: unimplemented paths `raise NotImplementedError` (or are clearly labeled
  placeholders) rather than silently returning fake results. Preserve this — don't replace a raise
  with a plausible-looking mock.
- **Citations are verified, never fabricated.** The project's stated principle is that every arXiv
  id / reference is cross-checked. When adding a citation in code comments or `references.bib`, it
  must be real and verified; flag unconfirmed ones rather than inventing them.
- The public API is re-exported from `stoa/__init__.py`; update `__all__` when adding exported names.
- **Paper numbers are checked, not trusted.** `scripts/check_paper_numbers.py` reads `experiments/*.json`,
  reformats each headline quantity, and greps `paper/*.tex` for it — the source, not the log or the PDF,
  since a log reports success for edits that matched nothing. Run it after any experiment re-run; it has
  already caught four double-rounding drifts (0.765 → "76" vs "77") and one claim that silently mixed two
  hyperparameter settings. When it fails, fix the paper, not the checker.
- **A fraction is reported with its denominator characterized.** Any "share of headroom captured" must
  be accompanied by the share a *causal* policy could reach (`sequential.attainable_ceiling`). Reporting
  the first without the second is how this project's central claim came to be an artifact for three
  revisions: 97% of the denominator was unreachable by construction.
- **A shared convention is fixed in every routine that uses it, or in none.** `place_with_beliefs`,
  `prefix_greedy_cost` and `_oracle_cost` all choose among equal-cost actions. Fixing the tie rule in one
  left the numerator and denominator of every reported fraction computed under different rules, and the
  only symptom was two artifacts quietly disagreeing. `tests/test_metric_fixes.py` pins their agreement.
- **A change that alters nothing is investigated, not celebrated.** A sliding-window filter added to
  `lrb.py` produced numbers identical to having no filter; the natural reading was "the window does not
  bind here" and it was wrong — a guard read `train_t[0]` on a list a later subsample had reordered.
- **A learned baseline asserts its own training signal.** `lrb.py` raises when label variance is zero.
  Without that, a truncation bug silently turned it into random eviction and it was reported as a
  measurement of LRB for two revisions.
- **Beliefs are compared on one magnitude scale.** `sequential.quantile_match` (with mid-rank tie
  handling) exists because the same policy with the same ordering scores +85% or −32% depending on the
  scale its beliefs sit on. State the convention or the number is not a quantity.
- `web/index.html` is a self-contained interactive visualization (no build step).
