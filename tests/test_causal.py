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
             "run_calibration_sensitivity.py", "run_split_sensitivity.py"]


@pytest.mark.parametrize("script", PRODUCERS)
def test_no_belief_magnitude_is_read_from_future_counts(script):
    """`fut` may feed the oracle and AUC truth; it may not set a policy's belief scale."""
    src = (SCRIPTS / script).read_text()
    bad = re.findall(r"^\s*(ref|ref_magnitudes|mean_reuse|reused)\s*=.*\bfut\b.*$", src, re.M)
    assert not bad, f"{script} sets a belief magnitude from future counts: {bad}"


def test_arrival_experiment_uses_one_split_and_windowed_labels():
    src = (SCRIPTS / "run_arrival_admission.py").read_text()
    assert "split_by_arrival(arrival, st)" in src, "train/eval cut must be the billing split"
    assert "windowed_training_labels(" in src, "training labels must end before the split"
    assert "analogue_magnitudes(" in src, "belief magnitudes must come from the pre-split period"
    assert not re.search(r"arrival\[len\(arrival\)\s*//\s*2\]", src), \
        "the median-arrival cut (a different instant from the billing split) is back"
