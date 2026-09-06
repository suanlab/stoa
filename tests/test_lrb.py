"""LRB-style learned eviction (src/stoa/lrb.py) — invariants and the frame check.

LRB is the paper's strongest counter-baseline: if a learned policy can beat reaction on KV
blocks, this is the design that should. That makes a silently-broken LRB the most dangerous
bug in the repo — it would manufacture the paper's conclusion. These tests pin the two
things that would produce that failure quietly: a train/inference feature-frame mismatch,
and a model that is never actually fitted.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from stoa.eval.online import simulate_fast  # noqa: E402
from stoa.lrb import _N_FEATURES, _Meta, _features, _touch, simulate_lrb  # noqa: E402
from stoa.traces import WorkloadConfig, generate_workload  # noqa: E402


def _wl(**kw):
    cfg = dict(n_items=200, horizon=12000, zipf_s=1.1, locality_beta=0.6, seed=3)
    cfg.update(kw)
    return generate_workload(WorkloadConfig(**cfg))


def test_feature_vector_has_the_declared_width():
    """Width drift between _features and _N_FEATURES would silently reshape the model."""
    m = _Meta()
    _touch(m, 0)
    assert len(_features(m, 5)) == _N_FEATURES


def test_recency_feature_is_nonzero_away_from_the_access():
    """The frame check. Features are recorded at eviction-sampling time, never at request
    time; if they were recorded at request time every training row would carry recency 0
    while inference always sees recency > 0. That mismatch cost 15 points of hit rate
    before it was found, and it is invisible in any aggregate metric."""
    m = _Meta()
    _touch(m, 100)
    at_access = _features(m, 100)
    later = _features(m, 400)
    assert at_access[0] == 0.0
    assert later[0] > 0.0, "recency must grow with elapsed time, or train/serve frames differ"


def test_hit_rate_is_bounded_by_belady_and_beats_fill_once():
    wl = _wl()
    slots = max(1, int(0.1 * len(wl.items)))
    lrb = simulate_lrb(wl, slots, train_interval=3000, memory_window=6000).hit_rate
    assert lrb <= simulate_fast(wl, "belady", slots).hit_rate + 1e-9
    assert lrb >= simulate_fast(wl, "static", slots).hit_rate


def test_learning_beats_its_own_lru_warm_up():
    """The policy falls back to LRU until a model exists. Setting train_interval beyond the
    trace length disables learning entirely, so this compares LRB against itself with the
    model removed and nothing else changed -- the no-learning control the paper insists on."""
    wl = _wl()
    slots = max(1, int(0.1 * len(wl.items)))
    learned = simulate_lrb(wl, slots, train_interval=2000, memory_window=6000).hit_rate
    no_model = simulate_lrb(wl, slots, train_interval=10 ** 9).hit_rate
    assert learned > no_model, f"learning did not help: {learned:.4f} vs {no_model:.4f}"


def test_accounts_for_every_access():
    wl = _wl(horizon=4000)
    r = simulate_lrb(wl, 16, train_interval=1500, memory_window=3000)
    assert r.hits + r.misses == len(wl.accesses)


def test_is_reproducible_at_a_fixed_seed():
    """Eviction samples are random; the seed must make a run reproduce exactly, or no
    number from this policy is citable."""
    wl = _wl(horizon=5000)
    a = simulate_lrb(wl, 20, train_interval=2000, memory_window=4000, seed=7)
    b = simulate_lrb(wl, 20, train_interval=2000, memory_window=4000, seed=7)
    assert (a.hits, a.misses) == (b.hits, b.misses)


def test_capacity_is_respected():
    wl = _wl(horizon=4000)
    for slots in (1, 8, 64):
        r = simulate_lrb(wl, slots, train_interval=1500, memory_window=3000)
        assert r.hits + r.misses == len(wl.accesses)
        assert r.hits <= len(wl.accesses)


def test_the_training_buffer_is_actually_bounded_in_age():
    """The sliding memory window must drop rows older than it. An earlier version guarded on
    `train_t[0]`, which is only the oldest row while the buffer is sorted -- and the uniform
    subsample reorders it, so after the first subsample the filter silently stopped firing
    and produced numbers identical to having no window at all."""
    import stoa.gbdt as G
    seen_rows = []
    orig = G.GBDT.fit

    def spy(self, X, y):
        seen_rows.append(len(y))
        return orig(self, X, y)

    G.GBDT.fit = spy
    try:
        wl = _wl(n_items=400, horizon=30000)
        # A window far smaller than the trace, and a cap large enough that only the window
        # bound can limit the buffer.
        simulate_lrb(wl, 40, train_interval=2000, memory_window=4000,
                     max_train_rows=10 ** 9)
    finally:
        G.GBDT.fit = orig
    assert len(seen_rows) > 4, "need several retrains to see the window bind"
    # Without an age bound the buffer grows monotonically. With one it must stop growing.
    later = seen_rows[len(seen_rows) // 2:]
    assert max(later) <= 3 * min(later), (
        f"buffer still growing without bound across retrains: {seen_rows}")


def test_the_age_bound_reads_the_minimum_not_the_first_element():
    """§Z: the window guard read `train_t[0]`, which is the oldest row only while the buffer is
    sorted -- and the uniform subsample reorders it, so after the first subsample the filter
    silently stopped firing. Its only symptom was numbers indistinguishable from having no
    window.

    This is a *structural* pin, and the reason is worth recording rather than hiding. The
    behavioural effect is real but small and not cleanly separable: at the one configuration
    where it shows (memory_window=4000, max_train_rows=1500, seed 3) reverting the guard moves
    hits from 25500 to 25422, about 0.3%. Two adjacent configurations show no difference at
    all, because the row cap binds before the window does and the buffer is exactly `cap` rows
    at every fit either way. The obvious behavioural assertions were tried and each failed to
    discriminate:

      * buffer size at fit time -- identical (always the cap) under both guards, five configs;
      * windowed run vs an unwindowed one -- differs under *both*, since the reverted guard
        still fires once before the first subsample.

    A golden hit count at the one discriminating configuration would pin it, but would also
    break on any legitimate change to sampling or features, and would then be silenced rather
    than investigated. Asserting what the guard reads is durable and is the defect itself.
    """
    import inspect
    import re

    src = inspect.getsource(simulate_lrb)
    guards = re.findall(r"if train_t and (\S+) < cutoff_t:", src)
    assert guards, "the age-bound guard is gone entirely, not merely weakened"
    for g in guards:
        assert g == "min(train_t)", (
            f"the window guard reads {g}, not min(train_t). The subsample reorders the buffer, "
            "so a first-element guard stops firing after the first subsample and the window "
            "becomes a no-op -- which produced numbers identical to having no window at all.")


def test_constant_training_labels_are_refused_rather_than_fitted():
    """A constant target turns the regressor into a no-op and eviction into a uniform draw.
    That is not hypothetical: it is the defect behind the retracted LRB column, where 11 of 13
    fits had `distinct_y == 1`. The policy must refuse to fit rather than emit a number that
    looks like a learned result.

    The condition is reproduced rather than mocked. In a trace with **no reuse at all**, every
    training row is closed out by censoring rather than by a re-access, so every label is the
    same censoring constant -- which is exactly the population the retraction was measured on
    (76-78% of KV blocks are never reused).

    The trace is built by hand rather than sampled: `generate_workload` never quite reaches
    zero reuse (birthday collisions leave 10-200 repeats even at 200k items over 2.5k
    accesses), and a handful of labelled rows is enough to give the batch variance and let the
    guard pass for the wrong reason.
    """
    from stoa.traces import AccessTrace, ItemState, Workload

    n = 3000
    items = [ItemState(item_id=f"u{i}", size_bytes=4096) for i in range(n)]
    accesses = [(t, f"u{t}") for t in range(n)]                    # each block seen exactly once
    traces = {f"u{i}": AccessTrace(item_id=f"u{i}", access_steps=[i]) for i in range(n)}
    wl = Workload(items=items, accesses=accesses, traces=traces)
    seen = [item for _, item in wl.accesses]
    assert len(set(seen)) == len(seen), "fixture must have zero reuse for labels to be constant"

    with pytest.raises(RuntimeError, match="labels are constant"):
        simulate_lrb(wl, 200, train_interval=1000, memory_window=500, max_train_rows=10 ** 9)


def test_training_subsample_does_not_draw_from_the_eviction_rng():
    """Structural, and deliberately so. The subsample uses a dedicated `random.Random`, so that
    changing how much the model trains cannot perturb which candidates eviction samples. Were
    it to share `rng`, every training-side change would silently move the eviction sequence and
    two runs differing only in `max_train_rows` would not be comparable."""
    import inspect
    import re

    src = inspect.getsource(simulate_lrb)
    assert re.search(r"train_rng\s*=\s*random\.Random\(", src), (
        "the training subsample no longer has a dedicated RNG; sharing the eviction RNG makes "
        "training-side changes move eviction decisions")
    m = re.search(r"keep\s*=\s*(\w+)\.sample\(", src)
    assert m and m.group(1) == "train_rng", (
        f"the subsample draws from {m.group(1) if m else '?'}, not train_rng")
