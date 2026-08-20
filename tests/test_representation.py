"""The representation axis (src/stoa/eval/representation.py).

This module exists to measure the paper's coupling claim, so the tests pin the properties
that would make the measurement meaningless rather than merely wrong: fact ids and questions
must survive a rewrite, the leak-free history split must survive it, and compression must be
cached so a fact is never rewritten differently in two places.
"""
from __future__ import annotations

import pytest

from stoa.eval.memqa import MemQAConfig, generate_task, split_history
from stoa.eval.representation import (MODES, compression_ratio, rewrite_task, token_cost)


def _task(**kw):
    cfg = dict(n_facts=8, n_questions=10, seed=1)
    cfg.update(kw)
    return generate_task(MemQAConfig(**cfg))


def _stub(prompt: str) -> str:
    """Deterministic 'compressor': returns two words regardless of input."""
    return "compressed fact"


def test_plaintext_is_the_identity():
    t = _task()
    assert rewrite_task(t, "plaintext") is t


def test_rewrite_preserves_ids_and_questions():
    """Questions reference fact ids. If a rewrite dropped or renamed one, every question
    needing it would become unanswerable and we would misread that as a utility loss."""
    t = _task()
    r = rewrite_task(t, "summary", compress_fn=_stub)
    assert set(r.facts) == set(t.facts)
    assert r.questions == t.questions


def test_rewrite_preserves_the_leak_free_history_split():
    """Losing `history_questions` silently reintroduces demand-signal leakage."""
    t = split_history(_task(n_questions=20), history_frac=0.5, seed=0)
    assert t.leak_free
    r = rewrite_task(t, "extract", compress_fn=_stub)
    assert r.leak_free
    assert r.history_questions == t.history_questions


def test_token_costs_follow_the_rewritten_text():
    t = _task()
    r = rewrite_task(t, "summary", compress_fn=_stub)
    for fid in r.facts:
        assert r.cost(fid) == token_cost(r.facts[fid]) == 2


def test_compression_ratio_reports_the_direction_of_change():
    t = _task()
    shrunk = rewrite_task(t, "summary", compress_fn=lambda p: "one")
    grown = rewrite_task(t, "summary", compress_fn=lambda p: "a b c d e f g h")
    assert compression_ratio(t, shrunk) < 1.0 < compression_ratio(t, grown)


def test_cache_is_used_so_a_fact_is_compressed_once():
    calls = []

    def counting(prompt: str) -> str:
        calls.append(prompt)
        return "x y"

    t = _task()
    cache: dict[str, str] = {}
    rewrite_task(t, "summary", compress_fn=counting, cache=cache)
    n_first = len(calls)
    rewrite_task(t, "summary", compress_fn=counting, cache=cache)
    assert n_first == len(t.facts)
    assert len(calls) == n_first, "second rewrite should be served entirely from cache"


def test_cache_keys_do_not_collide_across_modes():
    """A shared cache must not serve a 'summary' rewrite for an 'extract' request."""
    t = _task(n_facts=3)
    cache: dict[str, str] = {}
    seen = []
    rewrite_task(t, "summary", compress_fn=lambda p: seen.append("s") or "s s", cache=cache)
    rewrite_task(t, "extract", compress_fn=lambda p: seen.append("e") or "e", cache=cache)
    assert seen.count("s") == 3 and seen.count("e") == 3


def test_empty_compression_falls_back_to_the_original_text():
    """A model that returns nothing would delete evidence, and the loss would be blamed on
    the representation rather than on the failed call."""
    t = _task(n_facts=4)
    r = rewrite_task(t, "extract", compress_fn=lambda p: "   ")
    assert r.facts == t.facts


def test_unknown_mode_raises_rather_than_silently_passing_through():
    with pytest.raises(ValueError):
        rewrite_task(_task(), "latent", compress_fn=_stub)


def test_compressed_mode_requires_a_compressor():
    with pytest.raises(ValueError):
        rewrite_task(_task(), "summary")


def test_modes_are_the_documented_three():
    assert MODES == ("plaintext", "summary", "extract")
