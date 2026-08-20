"""Embedding-retrieval baselines. Offline — uses a deterministic fake embedder."""
import hashlib

from stoa.eval.memqa import MemQAConfig, evaluate, generate_task
from stoa.eval.retrieval import Retriever, cosine


def fake_embed(texts):
    """Deterministic hashed bag-of-words embedding — no network, but similarity is
    meaningful: texts sharing tokens get higher cosine than unrelated ones.
    Tokens are punctuation-stripped so 'E7' in a fact matches 'E7?' in a question."""
    import re

    vecs = []
    for t in texts:
        v = [0.0] * 64
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
            v[h % 64] += 1.0
        vecs.append(v)
    return vecs


def _task(seed=0):
    return generate_task(MemQAConfig(n_facts=20, n_questions=15, seed=seed))


def mock_reader(prompt: str) -> str:
    import re
    m = re.search(r"value of (E\d+)", prompt)
    if not m:
        return "?"
    fm = re.search(rf"\b{m.group(1)} = (\S+)", prompt)
    return fm.group(1) if fm else "?"


def test_cosine_basics():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_centrality_order_covers_all_facts():
    task = _task()
    r = Retriever(task, fake_embed, tag="test")
    order = r.centrality_order()
    assert sorted(order) == sorted(task.facts)      # a permutation, no drops


def test_per_query_retrieval_respects_budget():
    task = _task()
    r = Retriever(task, fake_embed, tag="test")
    r.embed_questions(task.questions)
    budget = 5 * task.tokens_per_fact
    for q in task.questions[:5]:
        keep = r.top_k_within_budget(q, budget)
        assert sum(task.cost(f) for f in keep) <= budget


def test_query_aware_retrieval_beats_random_coverage():
    """The RAG baseline should surface a question's own fact more often than chance."""
    task = _task()
    r = Retriever(task, fake_embed, tag="test")
    r.embed_questions(task.questions)
    budget = 5 * task.tokens_per_fact
    hits = sum(set(q.needed) <= r.top_k_within_budget(q, budget) for q in task.questions)
    rand = evaluate(task, mock_reader, budget, priority="random", seed=0)
    assert hits >= rand.n_answerable          # query-aware >= query-agnostic random


def test_evaluate_accepts_retriever_and_order():
    task = _task()
    r = Retriever(task, fake_embed, tag="test")
    r.embed_questions(task.questions)
    budget = 6 * task.tokens_per_fact
    res_r = evaluate(task, mock_reader, budget,
                     retriever=lambda q, b: r.top_k_within_budget(q, b))
    res_o = evaluate(task, mock_reader, budget, order=r.centrality_order())
    for res in (res_r, res_o):
        assert 0.0 <= res.accuracy <= 1.0
        assert res.calls == res.n_answerable
