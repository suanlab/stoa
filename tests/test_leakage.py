"""Guard against test-set leakage in the demand (heat) signal.

The STOA priority ranks facts by how often questions need them. If that count is
taken from the questions being SCORED, it reads the answer key — an oracle, not a
heat signal. These tests pin the leak-free contract.
"""
from stoa.eval.memqa import MemQAConfig, generate_task, split_history


def _task(seed=0):
    return generate_task(MemQAConfig(n_facts=30, n_questions=40, seed=seed))


def test_raw_task_is_not_leak_free():
    """A task without a history split computes demand from its own eval questions."""
    assert _task().leak_free is False


def test_split_history_is_leak_free_and_disjoint():
    t = split_history(_task(), history_frac=0.5, seed=0)
    assert t.leak_free is True
    hist = {(q.ask, q.answer) for q in t.history_questions}
    ev = {(q.ask, q.answer) for q in t.questions}
    assert hist and ev
    assert hist.isdisjoint(ev)                       # no shared question


def test_split_preserves_facts_and_costs():
    raw = _task()
    t = split_history(raw, seed=1)
    assert t.facts == raw.facts
    assert t.tokens_per_fact == raw.tokens_per_fact


def test_demand_uses_history_not_eval():
    """Demand must change when the history changes, and must not encode eval-only needs."""
    t = split_history(_task(), history_frac=0.5, seed=0)
    demand = t.demand()
    hist_needed = {f for q in t.history_questions for f in q.needed}
    # every fact with positive demand is needed by some HISTORY question
    assert {f for f, d in demand.items() if d > 0} <= hist_needed


def test_split_is_reproducible():
    a = split_history(_task(), seed=3)
    b = split_history(_task(), seed=3)
    assert [q.ask for q in a.questions] == [q.ask for q in b.questions]
    assert [q.ask for q in a.history_questions] == [q.ask for q in b.history_questions]


def test_per_question_outcomes_align_with_the_question_list():
    """The paired LoCoMo comparison differences arm A against arm B question by question. If
    `per_question` were shorter than the question list, or ordered differently, the pairing
    would silently compare unrelated questions and McNemar would report noise as signal.
    Unanswerable questions must therefore still occupy their slot as False."""
    from stoa.eval.memqa import MemQAConfig, evaluate, generate_task
    task = generate_task(MemQAConfig(n_facts=12, n_questions=15, seed=3))
    for budget in (0, 20, 10_000):
        r = evaluate(task, lambda p: "irrelevant reply", budget)
        assert len(r.per_question) == len(task.questions), (budget, len(r.per_question))
        assert sum(r.per_question) == r.n_correct
        # Every slot is filled even when the budget admits nothing at all.
        if budget == 0:
            assert not any(r.per_question)


def test_two_arms_produce_alignable_outcome_vectors():
    """Different context-selection arms must yield vectors of the same length in the same
    order, or the zip() that forms the discordant counts is meaningless."""
    from stoa.eval.memqa import MemQAConfig, evaluate, generate_task
    task = generate_task(MemQAConfig(n_facts=12, n_questions=15, seed=4))
    a = evaluate(task, lambda p: "x", 60, priority="stoa")
    b = evaluate(task, lambda p: "x", 60, priority="random", seed=1)
    assert len(a.per_question) == len(b.per_question) == len(task.questions)
