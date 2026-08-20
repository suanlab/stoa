# 1. Introduction

*(Draft prose for the CIDR paper. ~1.5 pp. Convert `[cite:key]` to `\cite{key}` at typesetting;
keys are in `docs/references.bib`.)*

Large language model (LLM) agents increasingly depend on memory that outlives a single context
window — the facts they extract, the turns they have seen, the tools they have run, the summaries they
have written. How well an agent answers, how fast it responds, and how much it costs are all decided by
one question asked of every memory item: **in what form should it be kept, where should it live, and
when should it be (re)materialized?** Concretely, each item can be stored as plaintext, a vector, a
graph, a latent/KV activation, or baked into parameters; it can sit on GPU, CPU, disk, or a remote tier;
and it can be written inline while serving, consolidated during an idle window, or left untouched.

Today this single decision is **fragmented across three research communities**, each of which optimizes
one axis and freezes the others. Memory-operating-system work — MemOS [cite:memos], MemGPT
[cite:memgpt] — provides the primitives to migrate an item between representations and tiers, but
*decides when to migrate by hand-tuned heuristics*. Reinforcement-learned memory managers — Memory-R1
[cite:memoryr1], Memory-as-Action [cite:memaction] — learn *what content* to add, update, or delete,
but operate over a single fixed representation and never choose a tier or schedule an offline write.
Systems-level cache tiering — LMCache [cite:lmcache] — places bytes across GPU/CPU/disk/remote to
minimize *miss rate or time-to-first-token*, with no notion of the downstream task utility that the
memory actually serves.

Our central observation is that **these three axes are coupled and cannot be optimized separately.**
Promoting an item to a latent or parameter representation is an expensive offline write, but it is then
cheap and fast to serve; leaving it as plaintext on disk is cheap to write but slow to read and heavy on
context tokens. The best representation depends on the tier and on how often the item will be reused;
the best tier depends on the representation's serving cost and the item's size; and whether to pay the
promotion cost at all depends on the budget. This is precisely a **budget-constrained scheduling
problem** — and a large body of ML-for-systems work (Belady-oracle imitation for cache replacement
[cite:parrot], GNN schedulers for clusters [cite:decima], offline-RL cache eviction in production
[cite:coldrl]) demonstrates that such problems are learnable.

The reason to act now is that every substrate exists but nothing orchestrates them: MemCube migration,
RL memory managers, tiered KV caches, latent memory [cite:mplus], and sleep-time consolidation
[cite:sleeptime] all landed by late 2025, unconnected. What is missing is a *learned scheduler* that,
per item and under a shared budget, trades accuracy against latency and cost, with **task utility — not
miss rate — as the objective.**

We present **STOA** (Storage & Tiered Orchestration for Agents), a learned control plane that decides
**representation × tier × timing jointly**, per memory item, as a single budget-constrained Markov
decision process. The operator sets a latency/cost/token budget `(L, C, B)` at serving time and one
policy adapts, tracing the accuracy–latency–cost frontier without retraining. To tame the combinatorics
(fifty legal actions per item, across many items) STOA factors the policy into representation, tier, and
timing heads over a graph network of the memory store, and evaluates it only over the legal joint
actions so masking is exact. It warm-starts the policy by imitating a Belady oracle and assigns credit
for delayed, query-time payoffs with a mix of oracle imitation and process supervision.

STOA is a research system, and this paper reports **early evidence on a trace-calibrated tiering
simulator and the LoCoMo benchmark** [cite:locomo] rather than a finished product. On that basis we
show: (i) the tier axis is a caching problem where a learned controller targets the gap between Belady
and LRU/H2O; (ii) a graph policy warm-started from a Belady oracle recovers essentially all of the
achievable placement-cost gap; (iii) budget-aware selection improves real LLM QA accuracy on LoCoMo by
30–58 points over budget-agnostic selection at tight budgets, while full-budget accuracy stays below
100% — real data exposes that both *placement* and *reasoning* matter. We also report a **negative
result** — policy-gradient fine-tuning does not beat the imitation warm-start on a *static* placement —
and then **resolve it**: reformulating the problem as a *sequential, occupancy-aware* MDP, in which the
policy observes live tier pressure, makes feasible near-optimal placement learnable, matching a greedy
oracle while a static policy overflows capacity. The lesson — that the missing ingredient was
occupancy *state*, not more optimization — is itself a design finding.

**Contributions.**
- **(C1)** A unifying formulation of agent-memory placement as a budget-constrained MDP over
  representation × tier × timing, with a single serve-time `(L, C, B)` knob (§3).
- **(C2)** The STOA control-plane architecture: a factored, masking-exact policy over a memory graph,
  with a Belady-imitation-plus-process-supervision credit-assignment recipe (§4).
- **(C3)** Early evidence on a calibrated simulator and LoCoMo (§5), including a negative result and its
  resolution via a sequential, occupancy-aware formulation (§5.6–5.7).
- **(C4)** A research agenda for a unified, budget-constrained, safety-aware memory runtime (§6).

We position STOA against prior work along the three axes it unifies in Table 1; no prior system occupies
their intersection — deciding representation, tier, *and* timing, learned, under a shared budget, with
task utility as the objective.
