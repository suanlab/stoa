"""The representation axis, exercised on real data.

The formulation in the paper decides representation x tier x timing jointly, and argues the
axes are coupled: a compressed representation is cheap to serve but may cost accuracy, so
the right choice depends on the budget. Every experiment so far has held representation
fixed at plaintext, which means the paper's own central claim about coupling has been
asserted and never measured. This module closes that gap on the utility side.

The mechanism is that `MemQATask` already separates a fact's *text* from its *token cost*.
A representation is therefore just a rewrite: same fact ids, same questions, same gold
answers, different text and a different cost. Accuracy at equal budget is then directly
comparable across representations, which is the quantity the formulation turns on.

Three representations, ordered by how much they throw away:

  plaintext  the turn verbatim -- the baseline every other experiment used
  summary    an LLM rewrite that keeps the content and drops the phrasing
  extract    an aggressive key-value reduction: entities and values only

What this can and cannot show. If accuracy per token were strictly better for a compressed
representation, the coupling claim would weaken -- a policy should then always compress, and
representation would not need to be decided per item. If instead the representations cross
over as the budget moves, the coupling is real and the joint decision is justified. Either
outcome is informative; the experiment is designed so both are reportable.

`compress_fn` is injected rather than imported so the whole path is testable offline
against a stub, the same convention `eval/memqa.py` uses for `answer_fn`.
"""
from __future__ import annotations

from collections.abc import Callable

from .memqa import MemQATask

# Prompts are fixed here rather than passed in, because a representation that changes
# between runs is not a representation -- it is a confound.
_PROMPTS: dict[str, str] = {
    "summary": (
        "Rewrite this line from a conversation as a single terse sentence. Keep every "
        "name, number, date and concrete detail. Drop pleasantries and phrasing. "
        "Reply with the rewrite only.\n\n{text}"
    ),
    "extract": (
        "Reduce this line from a conversation to its bare facts as "
        "'speaker: key=value; key=value'. Keep only names, numbers, dates, places and "
        "concrete nouns. No sentences. Reply with the reduction only.\n\n{text}"
    ),
}

MODES: tuple[str, ...] = ("plaintext", "summary", "extract")


def token_cost(text: str) -> int:
    """Word-count proxy, identical to the one `eval/locomo.py` uses for plaintext turns.

    Using the same proxy for every representation is what makes the budgets comparable. A
    real tokenizer would change all three costs together and so would not change the
    ordering this experiment measures.
    """
    return max(1, len(text.split()))


def rewrite_task(
    task: MemQATask,
    mode: str,
    compress_fn: Callable[[str], str] | None = None,
    cache: dict[str, str] | None = None,
) -> MemQATask:
    """Return `task` with every fact re-expressed in `mode`, ids and questions untouched.

    Question `needed` lists reference fact ids, so they survive the rewrite unchanged --
    which is the point: the same question is asked of the same evidence, expressed
    differently and costing differently. `cache` lets a caller reuse rewrites across budgets
    so the compression cost is paid once per fact, not once per budget.
    """
    if mode not in MODES:
        raise ValueError(f"unknown representation {mode!r}; choose from {MODES}")
    if mode == "plaintext":
        return task
    if compress_fn is None:
        raise ValueError(f"representation {mode!r} needs a compress_fn")

    cache = cache if cache is not None else {}
    facts: dict[str, str] = {}
    tokens: dict[str, int] = {}
    prompt = _PROMPTS[mode]
    for fid, text in task.facts.items():
        key = mode + "\x1f" + text   # unit separator: cannot occur in dialogue text
        if key not in cache:
            out = compress_fn(prompt.format(text=text)).strip()
            # A compression that returns nothing would silently delete evidence and show up
            # as a utility loss we would then misattribute to the representation.
            cache[key] = out or text
        facts[fid] = cache[key]
        tokens[fid] = token_cost(cache[key])
    return _rebuild(task, facts, tokens)


def _rebuild(task: MemQATask, facts: dict[str, str],
             tokens: dict[str, int]) -> MemQATask:
    """Rebuild a task with new fact text/costs, preserving the history split.

    Kept separate so the leak-free `history_questions` split cannot be dropped by accident:
    losing it would silently reintroduce the demand-signal leakage of `split_history`.
    """
    return MemQATask(facts, task.questions, task.tokens_per_fact, tokens,
                     task.history_questions)


def compression_ratio(original: MemQATask, rewritten: MemQATask) -> float:
    """Total tokens after / before. Below 1 means the representation actually compressed."""
    before = sum(original.cost(f) for f in original.facts)
    after = sum(rewritten.cost(f) for f in rewritten.facts)
    return after / before if before else 1.0
