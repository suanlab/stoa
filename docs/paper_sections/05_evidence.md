# 5. Preliminary Evidence

*(Draft prose. ~2 pp. All numbers come from reproducible artifacts in `experiments/`; regenerate figures
with `scripts/make_figures.py`. Numbers reflect a placeholder cost model pending real-trace calibration —
stated where relevant.)*

We evaluate STOA on a trace-calibrated tiering **simulator** and, for task utility, on the **LoCoMo**
benchmark [cite:locomo]. The simulator lets us establish oracle bounds and ablate each axis before a
real vLLM+LMCache deployment (future work, §6); it draws synthetic access traces with tunable popularity
skew and temporal locality, and its cost tables are illustrative placeholders to be fit from real
LMCache traces. We organize the evidence around three research questions: does joint learning help
(RQ1), can a Belady warm-start close the oracle gap (RQ2), and does one budget-conditioned policy trace
the frontier (RQ3)?

## 5.1 Offline references
For every workload we compute three reference points: a **Belady oracle** upper bound (the hindsight
per-item optimum, generally infeasible), a **capacity-feasible greedy** allocation (a strong achievable
reference), and a **naive** all-plaintext/CPU baseline. These bound every learned result from both
sides and anchor the ablations below.

## 5.2 Each axis is individually learnable (RQ1)
**Tier axis.** Placement onto a scarce hot tier over a cold backing store is a caching problem, and the
online policies that solve it — Belady (the offline optimum), LRU, and H2O/LFU [cite:h2o] — separate
only when the future differs from the past. On a stationary Zipf stream we reproduce the textbook
ordering **Belady > H2O > LRU**, with the Belady-over-heuristic gap *widening* as the hot tier grows
(+3.5 to +6.7 points as the cache goes from 5% to 20% of items; Fig 3-tier / Table 2). That gap is the
headroom a learned tier controller targets.

**Representation axis.** Holding the tier fixed and letting STOA choose representations cuts context-token
cost by **73%** versus keeping everything as plaintext, because hot items are promoted to token-cheap
forms. The two axes are complementary: the tier axis moves latency, the representation axis moves tokens.

## 5.3 A Belady warm-start recovers the placement gap (RQ2)
Following PARROT [cite:parrot], we warm-start the factored GNN policy by **behavioral cloning of the
capacity-feasible Belady placement**. On held-out workloads the cloned policy matches the oracle's
per-item action ~87% of the time and, in placement cost, recovers **essentially all** of the gap between
the naive baseline and the (capacity-feasible) Belady target — occasionally undercutting the greedy
target by generalizing to a cheaper feasible placement (Table 3). This establishes the imitation
warm-start the design calls for and sets up the credit-assignment question that RL fine-tuning addresses.

## 5.4 One knob traces the frontier (RQ3)
Given a serve-time token budget `B`, STOA's budget-conditioned controller returns the minimum-cost
placement that fits, and sweeping `B` traces a monotone frontier: as the budget tightens, more items are
promoted off plaintext and the consolidation cost rises smoothly (Fig 3). Crucially, **no single static
policy covers the range** — all-plaintext violates any tight budget, all-latent overspends on loose ones
— whereas one conditioned policy adapts across the whole frontier. (Under the current placeholder costs,
uniform-vector promotion is already efficient, so the *dollar* spread is modest; real-trace calibration
is expected to widen the regime where budget-conditioning pays off.)

## 5.5 Real task utility on LoCoMo (RQ1, real data)
The simulator assumes an item, once served, yields full utility; LoCoMo tests that assumption with a real
model. We treat each dialogue turn as a memory item and each question's `evidence` turns as the facts it
needs; under a context-token budget, only some turns fit, and we answer with gpt-4o-mini. Selecting turns
by evidence demand (STOA's heat signal) versus random selection improves QA accuracy by **30–58 points**
at tight budgets and converges at full budget (Fig 4). The gap is robust to the scoring metric: an LLM
judge (rather than substring match) yields the same +30–45-point advantage. Notably, **full-budget
accuracy plateaus near 50%, not 100%** — with all facts in context the model still errs — so on real data
both *placement* and *reasoning* limit utility, unlike our synthetic probe where a capable model reads
in-context facts perfectly. This is exactly the `U < 1` signal that motivates measuring task utility
directly rather than assuming it.

## 5.6 A clarifying negative result
Having warm-started the policy, we fine-tune it with REINFORCE (per-item credit, value-head baseline) on
the constrained objective (cost plus a tier-overflow penalty). On the **static** placement it does *not*
dominate the behavioral-cloning warm-start: raising the penalty weight drives capacity overflow toward
zero but only at a cost premium, and the cost/feasibility trade-off is noisy. The diagnosis is
structural — a policy that assigns all items *simultaneously* cannot see how much hot-tier capacity its
peers have already consumed, so it cannot learn *which* items to demote. The behavioral-cloning target
(the sequential greedy) already encodes that information; the static policy discards it.

## 5.7 …and its resolution: the occupancy-aware formulation
We therefore reformulate placement as a **sequential** MDP: items are placed one at a time in heat order,
the policy observes the *running tier occupancy*, and actions whose tier is full are masked out (the
remote tier is unbounded, so every rollout is feasible by construction). REINFORCE with return-to-go and
a value baseline now learns a policy that **reaches the greedy-Belady oracle cost and is feasible on
every trace** — a 6.7× improvement over a random-feasible policy — while the static behavioral-cloning
policy, at similar cost, *violates capacity on a third of the traces* (Fig 5). The missing ingredient was
occupancy *state*, not more optimization. We read this as evidence that the sequential, occupancy-aware
MDP is the right abstraction for the online setting we sketch in §6.
