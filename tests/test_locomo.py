"""LoCoMo adapter (arXiv:2402.17753). Offline — uses the cached data file, no network."""
import re

import pytest

from stoa.eval.memqa import _priority, evaluate, select_context

locomo = pytest.importorskip("stoa.eval.locomo")

try:
    TASKS = locomo.load_locomo(download=False)          # use cached data/locomo10.json
except FileNotFoundError:
    pytest.skip("data/locomo10.json not present; run the LoCoMo loader once to fetch it",
                allow_module_level=True)


def mock_reader(prompt: str) -> str:
    """Answer LoCoMo-style: echo the gold value if it appears in the shown context.
    (The real run uses an LLM; this validates the harness deterministically.)"""
    # Not used for correctness here — coverage-based tests use answerable counts.
    return ""


def test_tasks_well_formed():
    assert len(TASKS) == 10
    t = TASKS[0]
    assert len(t.facts) > 100 and len(t.questions) > 0
    for q in t.questions[:50]:
        assert set(q.needed) <= set(t.facts)            # evidence turns exist
        assert isinstance(q.answer, str) and q.answer


def test_per_fact_token_costs_present():
    t = TASKS[0]
    assert t.fact_tokens is not None
    assert all(t.cost(fid) >= 1 for fid in list(t.facts)[:20])


def test_demand_selection_beats_random_coverage():
    """RQ1 on real data: budget spent on needed turns answers far more questions."""
    t = TASKS[0]
    budget = int(0.3 * sum(t.fact_tokens.values()))
    stoa = select_context(t, _priority(t, "stoa", 0), budget)
    rand = select_context(t, _priority(t, "random", 0), budget)
    stoa_ans = sum(set(q.needed) <= stoa for q in t.questions)
    rand_ans = sum(set(q.needed) <= rand for q in t.questions)
    assert stoa_ans > rand_ans


def test_evaluate_only_calls_on_answerable():
    t = TASKS[0]
    calls = {"n": 0}

    def counting(_prompt):
        calls["n"] += 1
        return "unknown"

    budget = int(0.2 * sum(t.fact_tokens.values()))
    r = evaluate(t, counting, budget, priority="stoa")
    assert calls["n"] == r.n_answerable                 # no API call for evicted-evidence questions
    assert r.n_answerable <= r.n_questions
