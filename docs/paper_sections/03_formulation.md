# 3. Problem Formulation

*(Draft prose for the CIDR paper's technical core. Notation kept general; code
anchors in `src/stoa/environment.py`. Convert `$…$` to LaTeX at typesetting.)*

## 3.1 The per-item decision

An agent accumulates a set of memory items $\mathcal{M} = \{m_1, \dots, m_N\}$ —
facts, dialogue turns, tool results, summaries. For each item, a memory system must
answer three questions that are today decided separately:

- **Representation** $\rho \in \{\textsf{plaintext}, \textsf{vector}, \textsf{graph},
  \textsf{latent}, \textsf{parameter}\}$ — the semantic form the item is stored in,
  trading fidelity, serving-token cost, and write cost.
- **Tier** $\tau \in \{\textsf{GPU}, \textsf{CPU}, \textsf{disk}, \textsf{remote}\}$ —
  the physical location, trading read latency and capacity.
- **Timing** $t \in \{\textsf{online}, \textsf{sleep}, \textsf{noop}\}$ — *when* the
  placement is (re)materialized: inline on the serving path, deferred to an offline
  consolidation window, or not at all.

The central claim of this paper is that these three axes are **coupled** and must be
decided jointly. Promoting an item to a $\textsf{latent}$ or $\textsf{parameter}$
representation is an expensive offline write, but it is then cheap and fast to serve
(few or zero context tokens); leaving it as $\textsf{plaintext}$ on $\textsf{disk}$
is cheap to write but slow to read and token-heavy to serve. The optimal
representation therefore depends on the tier and on how often the item will be
reused, and the optimal tier depends on the representation's size and serving cost.
No axis can be optimized in isolation.

## 3.2 A budget-constrained MDP

We cast the orchestrator as a **budget-constrained Markov decision process** solved
by a learned policy $\pi_\theta$.

**State.** For each candidate item $m_i$ the state $s_t^{(i)}$ concatenates:
(a) *semantic* features — an embedding, recency, access frequency, a
provenance/version tag, and a predicted next-access time (a Belady-style forecast);
(b) *systems* features — the current representation and tier, the item size, and the
current occupancy/pressure of each tier; (c) *budget* features — the residual
latency, dollar, and token budget for the session; and (d) *query* context — the
current or predicted query embedding. Items are not independent: they are the nodes
of a **memory graph** whose edges encode co-access and entity links, which the policy
perceives jointly (§4).

**Action.** Each item is assigned one action
$a = (\rho, \tau, t) \in \mathcal{A}$. The raw product space has
$|\rho|\cdot|\tau|\cdot|t| = 5\cdot4\cdot3 = 60$ combinations, but some are
physically or semantically illegal and are **masked**: a $\textsf{parameter}$
promotion is expensive and therefore sleep-only (never inline $\textsf{online}$),
and $\textsf{latent}$/KV memory is meaningful only on the $\textsf{GPU}$/$\textsf{CPU}$
tiers. Masking removes $4 + 6 = 10$ combinations, leaving a fixed legal action set of
$|\mathcal{A}| = 50$ per item. Legality is defined once (`Action.is_legal`) and is
load-bearing: both the action enumeration and the policy's output distribution are
defined over exactly these 50 actions.

**Transition.** Executing $a$ realizes a cost tuple
$(\text{lat}, \text{cost}, \text{tok}, \text{fresh})$ from a transition model
calibrated on tiering traces. The timing axis sets the economics: a $\textsf{sleep}$
action amortizes the one-off promotion cost off the serving path; a $\textsf{noop}$
serves the item from its current form at no write cost.

**Objective.** We maximize expected downstream task utility subject to serve-time
budgets $L$ (latency), $C$ (dollars), and $B$ (context tokens):
$$
\max_{\theta}\ \mathbb{E}\!\left[\textstyle\sum_t U(a_t)\right]
\quad\text{s.t.}\quad
\text{Latency}\le L,\ \ \text{Cost}\le C,\ \ \text{Tokens}\le B .
$$
Here $U$ is a downstream reward — QA accuracy, or a step-level process reward — that
is realized only when the item is *used* to answer a query, many steps after the
placement is made.

**Lagrangian relaxation and the single knob.** We relax the constraints into the
per-step reward with learned duals $\lambda_L, \lambda_C, \lambda_B$:
$$
r_t \;=\; U \;-\; \lambda_L\,\text{lat} \;-\; \lambda_C\,\text{cost}
\;-\; \lambda_B\,\text{tok} \;+\; \beta\,\text{fresh}.
$$
The operator sets $(L, C, B)$ at serving time and the policy adapts, so a *single
control knob* moves the system along its accuracy–latency–cost frontier — the key
operational difference from a statically configured memory system, and the property
we probe as RQ3.

## 3.3 A factored policy over the memory graph

A joint head over 50 actions per item, across many items, is combinatorial. We
factor the policy over a shared trunk,
$$
\pi_\theta(a \mid s) \;\propto\; \pi_{\text{rep}}(\rho)\;\pi_{\text{tier}}(\tau)\;\pi_{\text{time}}(t),
$$
and — crucially — evaluate it **only over the 50 legal joint actions**: the three
factor logits are summed for each legal $a$ and softmaxed over that set. This keeps
the policy factored (cheap, three small heads) while making the induced distribution
*masking-exact* — it never places probability on an illegal combination. The trunk is
a graph network over the memory graph (§4), so co-accessed items inform one another's
placement rather than being decided independently.

## 3.4 Why this is learnable

The formulation is a scheduling problem with delayed, sparse, query-time reward and
edit-fractured trajectories — exactly the regime where imitation of a hindsight
(Belady) oracle plus process supervision has succeeded for cache replacement and
cluster scheduling. §4 instantiates the policy and its credit-assignment recipe; §5
reports early evidence that the tier and representation axes are individually
learnable and that a graph policy warm-started from a Belady oracle recovers almost
all of the achievable placement-cost gap — while also surfacing where the *static*
formulation of this MDP breaks down (§5.6, §6.1).
