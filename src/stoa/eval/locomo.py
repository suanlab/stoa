"""LoCoMo benchmark adapter (Maharana et al., 2024; arXiv:2402.17753).

LoCoMo is ~300-turn multi-session conversational memory: each sample is a long
two-speaker dialogue plus QA pairs whose `evidence` field names the dialogue turns
that support the answer. That maps directly onto the budget-QA harness (memqa.py):
    facts     = dialogue turns  (id = dia_id "D{session}:{turn}", text = "spk: text")
    questions = qa pairs        (needed = evidence turn ids, answer = gold answer)
Under a context-token budget only some turns fit, so budget-aware turn selection
(keep the turns questions actually need) beats random — a real-data RQ1 check.

Answers are scored by case-insensitive substring (a rough proxy; LoCoMo's official
metric is F1 / an LLM judge), so absolute accuracy is a lower bound — the STOA-vs-
random *gap* is the meaningful signal. Reuses stoa.llm for answers (NO GPU).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .memqa import MemQATask, Question

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_PATH = "data/locomo10.json"


def _download(path: str) -> None:
    import httpx

    r = httpx.get(LOCOMO_URL, timeout=120, follow_redirects=True)
    r.raise_for_status()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(r.text)


def _turns(conversation: dict):
    """Yield (dia_id, speaker, text) across all session_N turn lists, in order."""
    sessions = sorted((k for k in conversation if k.startswith("session_") and k[8:].isdigit()),
                      key=lambda k: int(k[8:]))
    for key in sessions:
        for turn in conversation[key]:
            if isinstance(turn, dict) and "dia_id" in turn:
                yield turn["dia_id"], turn.get("speaker", ""), turn.get("text", "")


def sample_to_task(sample: dict) -> MemQATask:
    """Convert one LoCoMo sample into a budget-QA task."""
    facts: dict[str, str] = {}
    fact_tokens: dict[str, int] = {}
    for did, spk, text in _turns(sample["conversation"]):
        facts[did] = f"{spk}: {text}"
        fact_tokens[did] = max(1, len(facts[did].split()))       # word-count token proxy

    questions: list[Question] = []
    for qa in sample.get("qa", []):
        ev = qa.get("evidence") or []
        ans = qa.get("answer")
        if ans is None or not ev:                                # skip unanswerable/adversarial
            continue
        needed = [e for e in ev if e in facts]
        if len(needed) != len(ev):                               # evidence turn missing -> skip
            continue
        questions.append(Question(str(qa["question"]), needed, str(ans)))
    return MemQATask(facts, questions, tokens_per_fact=20, fact_tokens=fact_tokens)


def load_locomo(path: str = DEFAULT_PATH, download: bool = True) -> list[MemQATask]:
    """Load LoCoMo as a list of budget-QA tasks (one per sample; downloads if absent)."""
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(f"{path} not found (set download=True to fetch from GitHub)")
        _download(path)
    data = json.loads(Path(path).read_text())
    return [sample_to_task(s) for s in data]
