"""Design invariants of the causal experiments (§AP).

The arrival-time experiment behind C2 passed every number check for months while (1) its training
labels counted evaluation-period accesses, (2) its train/eval cut and its billing split were
different instants, and (3) every policy's belief magnitudes came from the evaluation blocks' true
future counts. Number checks compare artifacts to the paper; they cannot see a design error that
produces a self-consistent wrong number. These tests pin the design itself.
"""
import re
from pathlib import Path

import pytest

from stoa.causal import (analogue_magnitudes, resample_to, split_by_arrival,
                         windowed_training_labels)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_split_uses_the_billing_boundary_convention():
    """An access at step `split` belongs to the evaluation period (future_counts uses s >= split)."""
    tr, ev = split_by_arrival([0, 4, 5, 6, 9], split=5)
    assert tr == [0, 1] and ev == [2, 3, 4]


def test_no_training_label_counts_an_access_at_or_after_the_split():
    arrival = [0, 2, 7, 9]
    ids = ["a", "b", "c", "d"]
    steps = {"a": [0, 3, 4, 20], "b": [2, 6, 11], "c": [7, 8], "d": [9, 10]}
    split, window = 10, 4
    idx, lab = windowed_training_labels(arrival, ids, steps, split, window)
    for i, v in zip(idx, lab):
        t = arrival[i]
        assert t + window < split, "a training row's window reaches the evaluation period"
        assert v == sum(1 for s in steps[ids[i]] if t < s <= t + window)
    # "c" (t=7) and "d" (t=9) cannot have a full window before the split and must be excluded.
    assert [ids[i] for i in idx] == ["a", "b"]
    # The access at step 20 (after the split) must not reach "a"'s label.
    assert lab[0] == 2


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        windowed_training_labels([0], ["a"], {"a": [0]}, 10, 0)


def test_resample_preserves_the_distribution():
    out = resample_to([1, 1, 1, 10], 8)
    assert len(out) == 8 and sorted(out) == out
    assert out.count(1.0) == 6 and out.count(10.0) == 2


def test_resample_refuses_an_empty_reference():
    with pytest.raises(ValueError):
        resample_to([], 3)


def test_analogue_ignores_blocks_not_yet_seen():
    """Blocks with prefix count 0 are not part of the previous period's population."""
    assert analogue_magnitudes([0, 0, 2, 4], 2) == [2.0, 4.0]


# --- structural: the leaks must not come back -------------------------------------------------

PRODUCERS = ["run_arrival_admission.py", "run_belief_controls.py", "run_capacity_ladder.py",
             "run_calibration_sensitivity.py", "run_split_sensitivity.py", "run_arrival_diagnostics.py"]


# Taint analysis on the AST (2026-10-19 re-verification). The regex this replaces matched one
# spelling of the original leak and missed realistic reintroductions: `future_counts(...)` called
# directly, a multi-line call, an alias (`truth = fut`), an annotated or tuple assignment, and the
# truth inlined into `quantile_match`. Sources are the evaluation future; taint propagates through
# assignments to a fixpoint; sinks are what set a policy's belief magnitude or training label.
import ast

SINK_NAMES = {"ref", "ref_magnitudes", "mean_reuse", "reused", "fit_lab"}
SINK_CALLS = {"analogue_magnitudes": None, "quantile_match": 1, "windowed_training_labels": None}


def _names(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n.id
        elif isinstance(n, ast.Attribute):
            yield n.attr


def _targets(t):
    if isinstance(t, ast.Name):
        yield t.id
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _targets(e)


def _pairs(n):
    """(target names, value node) pairs; `a, b = x, y` pairs element-wise, so `stats, fut =
    prefix_stats(st), future_counts(st)` taints `fut` and not `stats`."""
    tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
    for t in tgts:
        if (isinstance(t, (ast.Tuple, ast.List)) and isinstance(n.value, (ast.Tuple, ast.List))
                and len(t.elts) == len(n.value.elts)):
            for te, ve in zip(t.elts, n.value.elts):
                yield set(_targets(te)), ve
        else:
            yield set(_targets(t)), n.value


def belief_taint(src: str, sources=("fut", "future_counts")) -> list[str]:
    tree = ast.parse(src)
    tainted = set(sources)
    assigns = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value]
    changed = True
    while changed:
        changed = False
        for n in assigns:
            for tgts, val in _pairs(n):
                if tainted & set(_names(val)) and not tgts <= tainted:
                    tainted |= tgts
                    changed = True
    bad = []
    for n in assigns:
        for tgts, val in _pairs(n):
            if tgts & SINK_NAMES and tainted & set(_names(val)):
                bad.append(f"line {n.lineno}: {sorted(tgts & SINK_NAMES)} from "
                           f"{sorted(tainted & set(_names(val)))}")
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
            if fn in SINK_CALLS:
                pos = SINK_CALLS[fn]
                args = n.args if pos is None else n.args[pos:pos + 1]
                hit = tainted & {x for a in args for x in _names(a)}
                if hit:
                    bad.append(f"line {n.lineno}: {fn}() fed by {sorted(hit)}")
    return bad


@pytest.mark.parametrize("script", PRODUCERS)
def test_no_belief_magnitude_is_read_from_future_counts(script):
    """`fut` may feed the oracle, the ceiling and AUC truth; it may not set a belief or a label."""
    extra = ("y",) if script == "run_arrival_admission.py" else ()   # there `y` is the eval truth
    bad = belief_taint((SCRIPTS / script).read_text(), ("fut", "future_counts") + extra)
    assert not bad, f"{script}: {bad}"


@pytest.mark.parametrize("mutant", [
    "ref = [float(fut.get(i, 0)) for i in ev_ids]",                      # the original leak
    "ref = sorted(wl.future_counts(st).values())",                       # called directly
    "ref = analogue_magnitudes(\n    (fut[i] for i in ev_ids), n)",     # multi-line
    "truth = fut\nref = list(truth.values())",                            # alias
    "ref: list = [fut[i] for i in ev_ids]",                               # annotated
    "ref, other = [fut[i] for i in ev_ids], 0",                           # tuple target
    "stats, ref = wl.prefix_stats(st), wl.future_counts(st)",             # tuple, paired
    "b = quantile_match(scores, [fut[i] for i in ev_ids])",               # inlined into the sink
    "fit_lab = [y[i] for i in fit_idx]",                                  # labels from eval truth
])
def test_taint_check_catches_reintroduced_leaks(mutant):
    assert belief_taint(mutant, ("fut", "future_counts", "y")), f"missed: {mutant!r}"


def test_taint_check_allows_the_oracle_and_auc():
    ok = ("c_orac = place_with_beliefs(wl, {i: float(fut.get(i, 0)) for i in items})\n"
          "a_auc = auc(scores, [1.0 if y[i] > 0 else 0.0 for i in ev_idx])\n"
          "ref = analogue_magnitudes((v['count'] for v in pre.values()), n)\n"
          "b = quantile_match(scores, ref)\n"
          "stats, fut = wl.prefix_stats(st), wl.future_counts(st)\n"
          "ref_magnitudes = [v['count'] for v in stats.values()]\n")
    assert belief_taint(ok, ("fut", "future_counts", "y")) == []


def test_arrival_experiment_uses_one_split_and_windowed_labels():
    src = (SCRIPTS / "run_arrival_admission.py").read_text()
    assert "split_by_arrival(arrival, st)" in src, "train/eval cut must be the billing split"
    assert "windowed_training_labels(" in src, "training labels must end before the split"
    assert "analogue_magnitudes(" in src, "belief magnitudes must come from the pre-split period"
    assert not re.search(r"arrival\[len\(arrival\)\s*//\s*2\]", src), \
        "the median-arrival cut (a different instant from the billing split) is back"
