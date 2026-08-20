"""Embedding-retrieval baselines for the budget-QA harness (the RAG comparison).

Reviewers of a memory paper will ask the obvious question: *why not just retrieve
with embeddings?* Comparing STOA's demand-based selection only against RANDOM
selection is a straw man --- deployed systems rank memory by embedding similarity.
This module supplies two honest competitors, in two different regimes:

  * `centrality` (query-AGNOSTIC, same regime as STOA): rank turns by mean cosine
    similarity to the rest of the corpus and fill the budget with the most
    "representative" ones. This is the fair same-information comparison: like STOA,
    it commits to one context set before seeing any query.
  * `per-query top-k` (query-AWARE, the deployed-RAG regime): for each question,
    retrieve the most similar turns until the budget is exhausted. This baseline
    sees strictly more information than STOA (it knows the query), so it is
    expected to be strong --- we report it precisely because it is the honest bar.

Embeddings are injected as `embed_fn(texts) -> list[list[float]]` (so tests run
offline with a deterministic fake) and cached on disk to avoid repeat API cost.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

from .memqa import MemQATask, Question

CACHE_DIR = Path("data/emb_cache")


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return num / (na * nb)


def embed_cached(texts: list[str], embed_fn: Callable[[list[str]], list[list[float]]],
                 tag: str = "default") -> list[list[float]]:
    """Embed with an on-disk cache keyed by (tag, content hash)."""
    key = hashlib.sha256(("\n".join(texts)).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{tag}_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    vecs = embed_fn(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vecs))
    return vecs


class Retriever:
    """Embedding index over a task's facts, supporting both baseline regimes."""

    def __init__(self, task: MemQATask, embed_fn: Callable[[list[str]], list[list[float]]],
                 tag: str = "task"):
        self.task = task
        self.ids = list(task.facts)
        self.vecs = embed_cached([task.facts[i] for i in self.ids], embed_fn, f"{tag}_facts")
        self._qcache: dict[str, list[float]] = {}
        self._embed_fn = embed_fn
        self._tag = tag

    # --- query-agnostic: same regime as STOA (commit before seeing queries) ---
    def centrality_order(self) -> list[str]:
        """Rank facts by mean similarity to all other facts (most representative first)."""
        n = len(self.ids)
        scores = []
        for i in range(n):
            s = sum(cosine(self.vecs[i], self.vecs[j]) for j in range(n) if j != i)
            scores.append(s / max(n - 1, 1))
        return [self.ids[i] for i in sorted(range(n), key=lambda i: scores[i], reverse=True)]

    # --- query-aware: the deployed-RAG regime ---
    def embed_questions(self, questions: list[Question]) -> None:
        texts = [q.ask for q in questions]
        vecs = embed_cached(texts, self._embed_fn, f"{self._tag}_q")
        self._qcache = {q.ask: v for q, v in zip(questions, vecs)}

    def top_k_within_budget(self, question: Question, budget_tokens: int) -> set[str]:
        """Fill the budget with the turns most similar to this question."""
        qv = self._qcache.get(question.ask)
        if qv is None:
            qv = embed_cached([question.ask], self._embed_fn, f"{self._tag}_q1")[0]
            self._qcache[question.ask] = qv
        ranked = sorted(range(len(self.ids)), key=lambda i: cosine(qv, self.vecs[i]), reverse=True)
        keep: set[str] = set()
        used = 0
        for i in ranked:
            fid = self.ids[i]
            c = self.task.cost(fid)
            if used + c > budget_tokens:
                continue
            keep.add(fid)
            used += c
        return keep
