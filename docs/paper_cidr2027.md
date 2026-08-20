# STOA @ CIDR 2027 — Paper Skeleton (draft)

> **⚠️ SUPERSEDED (2026-08-09).** This skeleton planned a "learned control plane wins" paper for CIDR
> 2027, whose deadline (2026-08-04) has since passed. The empirical program it outlined was carried out
> and produced a different result: placement learnability does not hold up on production traces. The
> paper was reframed as a characterization — see `paper/` for the current source and `paper/README.md`
> for venue status, and `research_plan.md` for the full experimental record including the leaks and
> measurement bugs found along the way. Kept for provenance.

> **Target**: CIDR 2027 (submission **2026-08-04**; Jan 24–27, 2027, Amsterdam).
> **Fit**: CIDR welcomes bold, forward-looking data-systems ideas with *preliminary* evidence and honest
> open problems — matching STOA's current maturity (formulation + simulator + early real-LLM results).
> **Format caveat (verify on the CFP)**: CIDR is typically **non-anonymous (single-blind)**, uses its own
> LaTeX format, papers up to ~12 pp. Confirm length/format/anonymity before writing.
> **Sources of truth**: `docs/stoa_design.md`, `docs/research_plan.md`, `docs/survey.md`, `docs/references.bib`.
> Every result cited below maps to a reproducible artifact in `experiments/` (noted per section).

---

## Title (candidates)
1. **STOA: A Learned Control Plane for Tiered Agent Memory**
2. Where, How, and When to Remember: Budget-Constrained Orchestration of LLM Agent Memory
3. One Decision, Three Communities: Unifying Representation × Tier × Timing for Agent Memory

*Recommendation: Title 1 (systems-forward, names the artifact) with subtitle from Title 2.*

## Abstract (draft — ~180 words)
LLM agents increasingly depend on external memory, but the single decision that governs it — for each
memory item, in what **representation** (plaintext, vector, graph, latent/KV, parameter), on which
**storage tier** (GPU, CPU, disk, remote), and **when** to (re)materialize it (online vs. sleep-time) —
is today split across three communities that each optimize one axis and fix the others. Memory-OS work
migrates representations by heuristic; RL memory managers edit content over a single representation;
KV-cache tiering places bytes to cut miss-rate, ignoring task utility. We argue these axes are *coupled*
and form one **budget-constrained scheduling problem** that is learnable. We present **STOA**, a learned
control plane that decides representation × tier × timing jointly, per item, under latency/cost/token
budgets, as a single budget-constrained MDP exposing one serve-time **(L, C, B)** knob. STOA factors the
policy over a GNN of the memory graph and warm-starts it by imitating a Belady oracle. On a
trace-calibrated tiering simulator and the LoCoMo benchmark we show early evidence — and a clarifying
negative result that pinpoints the need for a *sequential, occupancy-aware* formulation — then lay out an
agenda for a unified, budget-constrained, safety-aware memory runtime.

---

## 1. Introduction  (~1.5 pp)
**Argument**: the governing decision of agent memory is one problem, fragmented into three.
- Hook: agents' quality/latency/cost hinge on *what memory is where, in what form, refreshed when*.
- The fragmentation (three axes / three communities):
  - **What representation** — MemOS (2507.03724), MemGPT (2310.08560): migration primitives exist, *scheduling is heuristic*.
  - **What content** — Memory-R1 (2508.19828), Memory-as-Action (2510.12635): RL over a *single* representation.
  - **Where physically** — LMCache (2510.09665): tiers bytes to minimize *miss-rate*, ignores task utility.
- **Key insight**: the axes are coupled (promoting to latent/parameter is expensive offline but cheap to
  serve; disk plaintext is cheap to write but slow/token-heavy to read) ⇒ a *single budget-constrained
  scheduling problem*, and ML-for-systems (PARROT 2006.16239, Decima 1810.01963, Cold-RL 2508.12485)
  shows such problems are learnable.
- **Contributions** (bulleted): (C1) a unifying formulation as a budget-constrained MDP over
  representation × tier × timing with a single (L,C,B) knob; (C2) the STOA control-plane architecture;
  (C3) a credit-assignment recipe (Belady imitation + process supervision + segment advantage);
  (C4) preliminary evidence on a calibrated simulator + LoCoMo, **plus a negative result** that scopes the
  next formulation.
- **Fig 1**: the three-axis decision and STOA sitting as a control plane over MemOS-store / LMCache-tiers /
  M+-latent path. **Table 1**: positioning matrix (from `research_plan.md §7`).

## 2. Motivation: Why Now  (~1 pp)
- All substrates landed by late 2025 but remain *unorchestrated*: MemCube migration, RL memory managers,
  tiered KV (LMCache), latent memory (M+ 2502.00592), sleep-time compute (2504.13171, ~5× savings).
- Worked coupling example (numbers from the cost model): plaintext = 0 write cost but ~800 tok/serve;
  latent = one-off promotion but 0 tok/serve — the right choice depends on reuse and budget.
- The missing piece: a *learned scheduler* that trades accuracy ↔ latency ↔ cost per item under a shared
  budget, with *task utility* (not miss-rate) as the objective.

## 3. Problem Formulation  (~1.5 pp)  — the technical core
- **State** `s_t` (per item): semantic (embedding, recency, freq, version/provenance, predicted next
  access) + systems (current repr, current tier, size, per-tier occupancy/pressure) + budget (residual
  L/C/B) + query context. *(cite design §2; code `environment.ItemState`)*
- **Action** (per item): Representation{5} × Tier{4} × Timing{3}; illegal combos masked
  (PARAMETER⇒sleep-only; LATENT⇒GPU/CPU) ⇒ **50 legal actions**. *(code `Action.space()`/`is_legal`)*
- **Objective**: maximize expected task utility s.t. Latency ≤ L, Cost ≤ C, Tokens ≤ B; **Lagrangian
  relaxation** `r = U − λ_L·lat − λ_C·cost − λ_B·tok + freshness` with learned duals ⇒ one serve-time
  **(L,C,B) knob**. *(code `environment.reward`)*
- **Why factored + graph**: 50 actions/item × many items ⇒ combinatorial; factor `π_rep·π_tier·π_time`
  on a shared GNN of the memory graph (Decima-style). *Masking-exact*: factor logits summed over the 50
  legal joint actions, softmaxed there.
- Emphasize: this is the contribution CIDR reviewers will scrutinize — keep it crisp and general.

## 4. STOA Architecture  (~1.5 pp)
- **Perception**: GNN over memory graph (nodes = items; edges = co-access/entity) + systems telemetry.
- **Policy**: factored actor (rep/tier/time heads, masked) + constrained critic (value head); separate
  admission/consolidation head for the sleep-time queue.
- **Execution**: online actions = MemCube migrate/fuse + LMCache put/get; offline (sleep-time) actions =
  promote plaintext→latent/parameter or graph-summarize during idle windows (LightMem-style, 2510.18866).
- **Credit assignment** (the hard part): (a) Belady-oracle imitation warm-start (PARROT); (b) process
  supervision (RAG-Gym 2502.13957); (c) segment advantage over edit-fractured trajectories (DCPO,
  2510.12635). Training: offline-RL first (safe on logs, Cold-RL-style) → guarded online.
- **Fig 2**: STOA block diagram (perception → factored policy → execution; online vs sleep-time paths).

## 5. Preliminary Evidence  (~2 pp)  — map each to an artifact
> Framing: "early evidence on a trace-calibrated *simulator* + a real benchmark; the simulator lets us
> establish oracle bounds and ablate axes before real vLLM+LMCache deployment (future work)."

- **5.1 Offline references** (`experiments/m0_oracle_baseline.json`): Belady oracle upper bound vs
  capacity-feasible greedy vs naive; establishes the ceiling every learned policy is measured against.
- **5.2 Per-axis ablation** — **Table 2** (`experiments/m3_ablation.json`):
  - *Tier axis (online caching)*: **Belady > H2O > LRU**; oracle gap widens with cache size
    (+3.5→+6.7 pp) — the headroom a learned tier controller targets.
  - *Representation axis (static)*: choosing representation cuts token cost **−73%** vs naive.
- **5.3 Learned controllers (RQ2)**:
  - Linear Belady-imitation controller (`experiments/m6_learned_controller.json`): under temporal
    locality beats both LRU and H2O (frequency+recency blend); honest null under stationary Zipf.
  - **GNN behavioral-cloning warm-start** (`experiments/m6_gnn_policy.json`): held-out imitation ~87%,
    recovers **~100%+** of the naive→(capacity-feasible-Belady) placement-cost gap. **Table 3**.
- **5.4 Budget controllability (RQ3)** — **Fig 3** (`experiments/m9_budget_frontier.json`): one knob B
  traces a feasible min-cost frontier; static baselines are single points that violate tight budgets or
  overspend on loose ones.
- **5.5 Real task utility on LoCoMo** — **Fig 4** (`experiments/eval_locomo.json`, arXiv:2402.17753):
  budget-aware (evidence-demand) turn selection vs random, gpt-4o-mini answers. STOA **+33–58 pp** at
  tight budgets; at full budget both ≈50% — real data exposes **U < 1** (vs 100% on the synthetic probe),
  i.e., placement *and* reasoning both matter. *(Note: substring metric = lower bound vs official F1/judge.)*
- **5.6 A clarifying negative result** (`experiments/m6_rl_finetune.json`): REINFORCE fine-tuning (per-item
  credit + value baseline) does **not** dominate the BC warm-start on the *static* placement — it can
  enforce feasibility only at a cost premium. Diagnosis: a *simultaneous* per-node policy lacks the
  tier-occupancy state a *sequential* placement uses. *(CIDR values this honesty — and we act on it next.)*
- **5.7 …and its fix — the occupancy-aware formulation** (`experiments/seq_placement.json`) — **Fig 5**:
  reformulated as a *sequential* MDP (place items in heat order; the policy observes running tier
  occupancy; unaffordable tiers are masked ⇒ feasible by construction), REINFORCE learns a policy that
  **reaches the greedy-Belady oracle cost (853)** and is **feasible on all traces**, vs random-feasible
  (5725, 6.7× worse) and static BC (868 but capacity-violating on 1/3). The missing ingredient was
  occupancy *state*, not more RL — turning §5.6 from a limitation into evidence for the right abstraction.

## 6. Open Problems / Research Agenda  (~1 pp)  — CIDR's favorite section
- **6.1 From feasible to *online***: §5.7 shows the *offline* sequential policy reaches the greedy oracle;
  the open problem is the *online* setting — placing under an unknown future query stream, with migration
  hysteresis and the sleep-time consolidation queue, where the oracle is no longer available.
- **6.2 Sim-to-real**: calibrate the simulator from real LMCache traces; validate on vLLM+LMCache
  (≤15% latency-prediction error gate). *(only GPU-gated step)*
- **6.3 Credit assignment under edit-fractured trajectories** (delayed, query-time payoff).
- **6.4 Safety & governance**: write-channel poisoning (MINJA 2503.03704, MPBench 2606.04329[재확인]),
  provenance guards, two-way unlearning (SBU 2602.17692[재확인]).
- **6.5 A unified Pareto benchmark**: accuracy × latency × $ × freshness × safety in one frame (today
  measured separately).

## 7. Related Work  (~0.75 pp)  — the three-axis framing, kept tight
- Memory-OS / representation; RL memory management / content; systems tiering & ML-for-systems.
- Emphasize STOA occupies the *intersection* none of them do (Table 1).

## 8. Conclusion  (~0.25 pp)
- One decision, three communities; STOA unifies them as a learned, budget-constrained control plane;
  early evidence is promising and the open problems define a runtime agenda.

---

## Figures & Tables checklist
| # | Content | Source artifact / status |
|---|---|---|
| Fig 1 | Three-axis decision + STOA control-plane overview | draw (schematic) — optional, Fig 2 may suffice |
| Fig 2 | STOA architecture block diagram | **done** (`scripts/make_figures.py::fig_architecture`) |
| Fig 3 | Budget-conditioned frontier (tokens↔$, feasibility) | `experiments/m9_budget_frontier.json` |
| Fig 4 | LoCoMo accuracy vs budget (STOA vs random) | `experiments/eval_locomo{,_judge}.json` (substring + LLM-judge) |
| Fig 5 | Sequential placement: RL vs random/BC/oracle | `experiments/seq_placement.json` |
| Table 1 | Positioning matrix (systems × decides-which-axis) | `research_plan.md §7` |
| Table 2 | Per-axis ablation (tier: Belady/H2O/LRU; repr: −73%) | `experiments/m3_ablation.json` |
| Table 3 | GNN BC warm-start: imitation acc + gap closed | `experiments/m6_gnn_policy.json` |

*Figures render via `python3 scripts/make_figures.py` → `paper/figs/*.{pdf,png}` (Okabe–Ito CVD-safe).*

## Citations (verified anchors — keys in `docs/references.bib`)
MemOS 2507.03724 · MemGPT 2310.08560 · Memory-R1 2508.19828 · Memory-as-Action/DCPO 2510.12635 ·
LMCache 2510.09665 · M+ 2502.00592 · Sleep-time 2504.13171 · LightMem 2510.18866 · RAG-Gym 2502.13957 ·
PARROT 2006.16239 · H2O 2306.14048 · Cold-RL 2508.12485 · Decima 1810.01963 · LoCoMo 2402.17753 ·
LongMemEval 2410.10813 · MemoryAgentBench 2507.05257 · MINJA 2503.03704.
*Keep the verification discipline: `[재확인]` items (e.g., MPBench 2606.04329, SBU 2602.17692) need an
independent ID re-check before camera-ready; do not cite the unconfirmed ones (Adaptive-RAG 2403.14403,
Mem-α 2509.25911) without verification.*

---

## Before submission — what to strengthen (priority order, ~3 weeks)
1. **Sharpen the formulation (§3)** — this is the reviewed core; make it self-contained and general.
2. **Regenerate camera-quality figures** from the JSON artifacts (Fig 3, Fig 4; Tables 2–3).
3. **LoCoMo: widen the run** — more samples/questions and, if time, an LLM-judge or F1 metric so absolute
   numbers aren't only a substring lower bound (the *gap* already holds).
4. **Frame the negative result as a contribution** (§5.6→§6.1), not an apology.
5. ~~Optional stretch: sequential occupancy-aware controller~~ **DONE (§5.7, Fig 5)** — sequential-RL
   reaches the greedy oracle and dominates static BC. Optionally extend toward the *online* setting (§6.1).
6. Honesty pass: label all simulator numbers as placeholder-cost-model results pending real-trace
   calibration; keep citation `[재확인]`/`[검증 필요]` discipline.

## Division of the ~12 pages (rough)
Intro 1.5 · Motivation 1 · Formulation 1.5 · Architecture 1.5 · Evidence 2 · Open Problems 1 ·
Related 0.75 · Conclusion 0.25 · (refs/figures fill the rest).
