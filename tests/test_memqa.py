"""M-eval: LLM-backed memory-QA under budget (RQ1). Uses a mock reader — no network."""
import re

from stoa.eval.memqa import (
    MemQAConfig,
    compare,
    evaluate,
    generate_multihop_task,
    generate_task,
    select_context,
)


def mock_reader(prompt: str) -> str:
    """Perfect reader/reasoner: answers correctly iff the needed facts are in context."""
    m = re.search(r"value of (E\d+)", prompt)          # single-hop
    if m:
        e = m.group(1)
        fm = re.search(rf"\b{e} = (\S+)", prompt)
        return fm.group(1) if fm else "unknown"
    m = re.search(r"does (A\d+) lead to", prompt)       # two-hop: A -> B, then B = v
    if m:
        a = m.group(1)
        lm = re.search(rf"{a} -> (B\d+)", prompt)
        if lm:
            b = lm.group(1)
            vm = re.search(rf"\b{b} = (\S+)", prompt)
            return vm.group(1) if vm else "unknown"
    return "unknown"


def _task(seed=0):
    return generate_task(MemQAConfig(n_facts=40, n_questions=60, zipf_s=1.1, seed=seed))


def test_task_reproducible():
    a, b = _task(3), _task(3)
    assert a.facts == b.facts
    assert [(q.ask, q.answer) for q in a.questions] == [(q.ask, q.answer) for q in b.questions]


def test_select_context_respects_budget():
    task = _task()
    keep = select_context(task, list(task.facts), budget_tokens=10 * task.tokens_per_fact)
    assert len(keep) == 10


def test_perfect_reader_accuracy_equals_answerable_rate():
    task = _task()
    r = evaluate(task, mock_reader, 15 * task.tokens_per_fact, priority="stoa")
    assert r.n_correct == r.n_answerable            # perfect reader answers all answerable
    assert r.calls == r.n_answerable                # no API call for evicted facts
    assert abs(r.accuracy - r.answerable_rate) < 1e-9


def test_tighter_budget_lowers_coverage():
    task = _task()
    loose = evaluate(task, mock_reader, 30 * task.tokens_per_fact, priority="stoa")
    tight = evaluate(task, mock_reader, 5 * task.tokens_per_fact, priority="stoa")
    assert tight.answerable_rate <= loose.answerable_rate


def test_stoa_heat_beats_random_at_equal_budget():
    """RQ1: spending the budget on hot facts answers more questions than random."""
    wins = 0
    for seed in range(5):
        task = _task(seed)
        budget = 10 * task.tokens_per_fact
        s = evaluate(task, mock_reader, budget, priority="stoa", seed=seed)
        r = evaluate(task, mock_reader, budget, priority="random", seed=seed)
        wins += s.accuracy >= r.accuracy
    assert wins == 5


def test_multihop_needs_both_facts():
    """Two-hop: a chain is answerable only when BOTH its facts fit the budget."""
    task = generate_multihop_task(MemQAConfig(n_facts=20, n_questions=40, seed=1))
    assert all(len(q.needed) == 2 for q in task.questions)
    full = evaluate(task, mock_reader, 40 * task.tokens_per_fact, priority="stoa")
    assert full.n_answerable == full.n_questions    # budget fits all 2N facts
    assert full.n_correct == full.n_answerable       # perfect reasoner


def test_multihop_stoa_beats_random():
    task = generate_multihop_task(MemQAConfig(n_facts=20, n_questions=40, seed=2))
    budget = 12 * task.tokens_per_fact
    s = evaluate(task, mock_reader, budget, priority="stoa", seed=2)
    r = evaluate(task, mock_reader, budget, priority="random", seed=2)
    assert s.accuracy >= r.accuracy


def test_compare_returns_frontiers():
    task = _task()
    fr = compare(task, mock_reader, [b * task.tokens_per_fact for b in (5, 15, 40)])
    assert set(fr) == {"stoa", "random"} and len(fr["stoa"]) == 3
