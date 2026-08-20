"""LLM-backed memory-QA under a token budget (docs/research_plan.md §5, §8 RQ1).

Grounds the reward's task-utility term U (so far hardcoded to 1.0) in a real LLM:
under a context-token budget B, only some memory facts fit, so questions whose
supporting facts are evicted fail. Selecting which facts to keep by STOA's heat
signal (fact demand) should beat a budget-agnostic random selection at equal B.

Two task families:
  * single-hop (`generate_task`)      — "E{i} = v"; one fact per question. Easy: a
    capable model reads it off, so accuracy ~ in-context rate (validates the loop).
  * two-hop  (`generate_multihop_task`) — "A{i} -> B{j}" + "B{j} = v"; the answer
    needs BOTH facts AND a reasoning step, with the many "B = v" facts as
    distractors. This exposes the model's own utility U<1 and rewards co-placing
    linked facts (the structure the GNN co-access graph captures).

The LLM is injected as `answer_fn(prompt) -> str`, so the harness is fully testable
with a mock (no network) and runs against any OpenAI-compatible model via
`stoa.llm.OpenAIClient(...).answer_fn()`. NO GPU.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class Question:
    ask: str                 # the natural-language question
    needed: list[str]        # fact ids that must ALL be in context to answer
    answer: str              # expected value (scored by case-insensitive substring)


@dataclass
class MemQAConfig:
    n_facts: int = 40        # single-hop: number of facts; multi-hop: number of chains
    n_questions: int = 60
    zipf_s: float = 1.1
    tokens_per_fact: int = 6
    seed: int = 0


@dataclass
class MemQATask:
    facts: dict[str, str]                # id -> fact text
    questions: list[Question]            # the EVALUATION questions
    tokens_per_fact: int                 # default per-fact cost (synthetic tasks)
    fact_tokens: dict[str, int] | None = None   # per-fact cost (real tasks, e.g. LoCoMo turns)
    history_questions: list[Question] | None = None  # PAST questions used to estimate demand

    def cost(self, fid: str) -> int:
        if self.fact_tokens is not None:
            return self.fact_tokens.get(fid, self.tokens_per_fact)
        return self.tokens_per_fact

    def demand(self) -> dict[str, int]:
        """Per-fact demand = how often it is needed by the HISTORY questions.

        CRITICAL: demand must be estimated from *past* questions, never from the
        questions being evaluated — reading `needed` off the evaluation set is
        test-set leakage (it is the answer key), which makes the heat signal an
        oracle. Use `split_history()` to build a leak-free task. If no history is
        set we fall back to `self.questions`, which is ONLY valid when the task is
        explicitly a history/oracle probe.
        """
        source = self.history_questions if self.history_questions else self.questions
        d: Counter = Counter()
        for q in source:
            for fid in q.needed:
                d[fid] += 1
        return {fid: d.get(fid, 0) for fid in self.facts}

    @property
    def leak_free(self) -> bool:
        """True when demand is estimated from questions disjoint from the eval set."""
        if not self.history_questions:
            return False
        evalset = {(q.ask, q.answer) for q in self.questions}
        return not any((h.ask, h.answer) in evalset for h in self.history_questions)


def _nonce(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5))


def _zipf_questions(rng: random.Random, keys: list[str], n: int, s: float) -> list[str]:
    order = keys[:]
    rng.shuffle(order)
    weights = [1.0 / ((r + 1) ** s) for r in range(len(order))]
    return [rng.choices(order, weights=weights, k=1)[0] for _ in range(n)]


def generate_task(cfg: MemQAConfig | None = None) -> MemQATask:
    """Single-hop: `E{i} = {nonce}`, questions ask one entity's value."""
    cfg = cfg or MemQAConfig()
    rng = random.Random(cfg.seed)
    facts = {f"E{i}": "" for i in range(cfg.n_facts)}
    answer = {}
    for i in range(cfg.n_facts):
        v = _nonce(rng)
        facts[f"E{i}"] = f"E{i} = {v}"
        answer[f"E{i}"] = v
    asked = _zipf_questions(rng, [f"E{i}" for i in range(cfg.n_facts)], cfg.n_questions, cfg.zipf_s)
    questions = [Question(f"What is the value of {e}?", [e], answer[e]) for e in asked]
    return MemQATask(facts, questions, cfg.tokens_per_fact)


def generate_multihop_task(cfg: MemQAConfig | None = None) -> MemQATask:
    """Two-hop: `A{i} -> B{i}` + `B{i} = {nonce}`; answer needs both facts + a hop.
    The full set of `B = v` facts are distractors for the link-following step."""
    cfg = cfg or MemQAConfig()
    rng = random.Random(cfg.seed)
    facts: dict[str, str] = {}
    answer: dict[str, str] = {}
    for i in range(cfg.n_facts):
        v = _nonce(rng)
        facts[f"link{i}"] = f"A{i} -> B{i}"
        facts[f"val{i}"] = f"B{i} = {v}"
        answer[f"chain{i}"] = v
    asked = _zipf_questions(rng, [f"chain{i}" for i in range(cfg.n_facts)], cfg.n_questions, cfg.zipf_s)
    questions = [
        Question(f"Following the links, what value does A{c[5:]} lead to?",
                 [f"link{c[5:]}", f"val{c[5:]}"], answer[c])
        for c in asked
    ]
    return MemQATask(facts, questions, cfg.tokens_per_fact)


def split_history(task: MemQATask, history_frac: float = 0.5, seed: int = 0,
                  strict: bool = True) -> MemQATask:
    """Split questions into a (history, evaluation) pair — the leak-free setup.

    Demand (STOA's heat signal) is estimated from the history questions; accuracy is
    measured on the held-out evaluation questions. This is the memory-system analogue
    of "past access frequency predicts future access", and the only honest way to use
    the `needed` annotations: never read them off the questions being scored.

    `strict=True` (default) additionally drops evaluation questions that are *identical*
    to a history question, so the measurement is generalization to UNSEEN questions.
    Set `strict=False` to model a repeat-access workload (where re-asking is legitimate
    and is exactly the reuse signal caching exploits) — but then the demand signal
    trivially knows the repeated questions' evidence, so report it as such.
    """
    qs = list(task.questions)
    random.Random(seed).shuffle(qs)
    cut = max(1, int(len(qs) * history_frac))
    history, evaluation = qs[:cut], qs[cut:]
    if strict:
        seen = {(q.ask, q.answer) for q in history}
        evaluation = [q for q in evaluation if (q.ask, q.answer) not in seen]
    if not evaluation:                                   # degenerate split guard
        history, evaluation = qs[:-1], qs[-1:]
    return MemQATask(task.facts, evaluation, task.tokens_per_fact, task.fact_tokens,
                     history_questions=history)


def select_context(task: MemQATask, priority: list[str], budget_tokens: int) -> set[str]:
    """Include facts in `priority` order until the token budget is exhausted.

    Uses per-fact token costs when the task provides them (real turns vary in
    length); skips a too-big fact and keeps trying smaller ones down the list.
    """
    keep: set[str] = set()
    used = 0
    for fid in priority:
        c = task.cost(fid)
        if used + c > budget_tokens:
            continue
        keep.add(fid)
        used += c
    return keep


def _priority(task: MemQATask, kind: str, seed: int) -> list[str]:
    if kind == "stoa":                                   # heat = fact demand across questions
        demand = task.demand()
        return sorted(task.facts, key=lambda f: demand.get(f, 0), reverse=True)
    if kind == "random":
        r = random.Random(seed)
        order = list(task.facts)
        r.shuffle(order)
        return order
    raise ValueError(f"unknown priority {kind!r}")


@dataclass
class QAResult:
    priority: str
    budget_tokens: int
    n_questions: int
    n_answerable: int          # questions whose supporting facts all fit the budget
    n_correct: int
    calls: int
    context_facts: int = 0
    # Per-question outcome in task order: True iff answered correctly. A question whose
    # supporting facts did not fit the budget is False without an API call. Recorded because
    # every arm answers the SAME questions, which makes the comparison PAIRED -- and a
    # paired comparison needs per-question outcomes, not just totals (see stoa.stats.mcnemar_exact).
    per_question: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_questions if self.n_questions else 0.0

    @property
    def answerable_rate(self) -> float:
        return self.n_answerable / self.n_questions if self.n_questions else 0.0

    @property
    def reasoning_acc(self) -> float:
        """Accuracy *given* the facts were in context (isolates the model's U)."""
        return self.n_correct / self.n_answerable if self.n_answerable else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accuracy"] = round(self.accuracy, 4)
        d["answerable_rate"] = round(self.answerable_rate, 4)
        d["reasoning_acc"] = round(self.reasoning_acc, 4)
        return d


def _substring_correct(gold: str, reply: str) -> bool:
    return gold.lower() in reply.lower()


def evaluate(
    task: MemQATask,
    answer_fn: Callable[[str], str],
    budget_tokens: int,
    priority: str = "stoa",
    seed: int = 0,
    judge_fn: Callable[[str, str, str], bool] | None = None,
    order: list[str] | None = None,
    retriever: Callable[[Question, int], set[str]] | None = None,
) -> QAResult:
    """Answer each question with only the budgeted facts in context; score by value match.

    Two regimes:
      * global (default): one context set is chosen for ALL questions, by `priority`
        (or an explicit `order`) — the placement regime STOA operates in.
      * query-aware: pass `retriever(question, budget) -> fact ids` to select context
        per question (the deployed-RAG regime; see eval/retrieval.py). This baseline
        sees the query, so it is strictly better informed than the global regime.

    Questions whose supporting facts are absent are wrong without an API call
    (the model cannot know an unshown fact) — bounding cost to answerable questions.
    Scoring is case-insensitive substring by default; pass `judge_fn(question, gold,
    reply) -> bool` (e.g. an LLM judge) for a more faithful metric.
    """
    global_order = order if order is not None else _priority(task, priority, seed)
    global_keep = select_context(task, global_order, budget_tokens)

    n_ans = n_correct = calls = 0
    kept_sizes: list[int] = []
    outcomes: list[bool] = []
    for q in task.questions:
        if retriever is not None:
            keep = retriever(q, budget_tokens)
            ctx_ids = [f for f in global_order if f in keep]
        else:
            keep = global_keep
            ctx_ids = [f for f in global_order if f in keep]
        kept_sizes.append(len(keep))
        if not set(q.needed) <= keep:
            outcomes.append(False)
            continue
        n_ans += 1
        context = "\n".join(task.facts[fid] for fid in ctx_ids)
        prompt = f"Known facts:\n{context}\n\nQuestion: {q.ask} Reply with only the value."
        reply = answer_fn(prompt)
        calls += 1
        correct = judge_fn(q.ask, q.answer, reply) if judge_fn else _substring_correct(q.answer, reply)
        outcomes.append(bool(correct))
        if correct:
            n_correct += 1
    ctx_facts = round(sum(kept_sizes) / len(kept_sizes)) if kept_sizes else 0
    return QAResult(priority, budget_tokens, len(task.questions), n_ans, n_correct, calls,
                    ctx_facts, outcomes)


def compare(task: MemQATask, answer_fn: Callable[[str], str], budgets: list[int],
            seed: int = 0, judge_fn: Callable[[str, str, str], bool] | None = None
            ) -> dict[str, list[QAResult]]:
    """Accuracy-vs-budget frontier for STOA-heat vs random fact selection."""
    return {p: [evaluate(task, answer_fn, b, priority=p, seed=seed, judge_fn=judge_fn) for b in budgets]
            for p in ("stoa", "random")}
