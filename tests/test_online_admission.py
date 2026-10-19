"""Online admission control (scripts/run_online_admission.py, H6).

Pins three things: the controller's arithmetic on a workload small enough to bill by hand; that the
per-access bill of a STATIC assignment is exactly `place_with_beliefs` (tier-only), so the online
and static numbers are in one frame; and that the script feeds no belief from the future.
"""
import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_online_admission.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oa = _load(SCRIPT, "run_online_admission")

from stoa.environment import Representation, Tier  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS as W  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

SIM = TieringSimulator()
CM = oa.cost_model(SIM, W, oa.tier_only_actions(), Representation.PLAINTEXT)
GPU, CPU, REM, DSK = 0, 1, 2, 3                       # fast -> slow
ONE_EACH = {Tier.GPU: 1.0, Tier.CPU: 1.0, Tier.REMOTE_RDMA: 1.0, Tier.DISK: float("inf")}


def _per(t):
    return W.lambda_latency * SIM.read_latency(t) + W.lambda_tokens * 800


def test_cost_model_is_read_off_the_simulator():
    assert CM["tiers"] == [Tier.GPU, Tier.CPU, Tier.REMOTE_RDMA, Tier.DISK]
    for t in CM["tiers"]:
        assert CM["per"][t] == pytest.approx(_per(t))     # 0.42, 0.6, 1.0, 1.6
        assert CM["fixed"][t] == 0.0
    assert oa.migration_cost(Tier.CPU, Tier.REMOTE_RDMA, CM["read_ms"], W, "read+write") \
        == pytest.approx(0.1 * (2.0 + 6.0))
    assert oa.migration_cost(Tier.CPU, Tier.REMOTE_RDMA, CM["read_ms"], W, "zero") == 0.0


# Hand-built workload. One slot each in GPU / CPU / REMOTE, DISK unbounded.
#   pre-split: block 0 (belief 3) -> GPU, block 1 (belief 1) -> CPU at the split.
#   new:       block 2 (belief 2), block 3 (belief 1), block 4 (belief 0.5).
#   post-split accesses: 2 0 3 1 2 4 3
# new-behind:
#   2 arrives: REMOTE has room, but CPU's resident (1, belief 1) ranks behind 2 -> 2 takes CPU,
#     1 is demoted CPU->REMOTE.                                           billed CPU    0.6
#   0: GPU 0.42.  3 arrives: only DISK has room; GPU (3), CPU (2) outrank it; REMOTE holds 1 at
#     belief 1 = a tie, which a new block loses -> 3 to DISK.               billed DISK   1.6
#   1: REMOTE 1.0.  2: CPU 0.6.  4 (0.5) outranks nobody -> DISK 1.6.  3: DISK 1.6.
#   total 0.6+0.42+1.6+1.0+0.6+1.6+1.6 = 7.42, one displacement, one move CPU->REMOTE.
# new-ahead: identical until 3, which now wins the tie with 1: 3 -> REMOTE (1.0), 1 -> DISK, so
#   1's access is billed 1.6 and 3's second access 1.0; 4 still loses to 3 (0.5 < 1).
#   total 0.6+0.42+1.0+1.6+0.6+1.6+1.0 = 6.82, two displacements, moves CPU->REMOTE, REMOTE->DISK.
BELIEFS = [3.0, 1.0, 2.0, 1.0, 0.5]
IS_NEW = [False, False, True, True, True]
EVENTS = [2, 0, 3, 1, 2, 4, 3]


@pytest.mark.parametrize("policy", ["onestep", "cascade"])
def test_hand_workload_new_loses_ties(policy):
    r = oa.run_online(BELIEFS, IS_NEW, 1.0, ONE_EACH, CM, EVENTS, new_ahead=False, policy=policy)
    assert r["access_cost"] == pytest.approx(7.42)
    assert r["n_displacements"] == 1
    assert r["moves"] == Counter({(CPU, REM): 1})
    assert r["where"] == [GPU, REM, CPU, DSK, DSK]
    assert oa.total_migration(r["moves"], CM, W, "read+write") == pytest.approx(0.8)


@pytest.mark.parametrize("policy", ["onestep", "cascade"])
def test_hand_workload_new_wins_ties(policy):
    r = oa.run_online(BELIEFS, IS_NEW, 1.0, ONE_EACH, CM, EVENTS, new_ahead=True, policy=policy)
    assert r["access_cost"] == pytest.approx(6.82)
    assert r["n_displacements"] == 2
    assert r["moves"] == Counter({(CPU, REM): 1, (REM, DSK): 1})
    assert r["where"] == [GPU, DSK, CPU, REM, DSK]
    assert oa.total_migration(r["moves"], CM, W, "read+write") == pytest.approx(0.8 + 1.8)


def test_hand_workload_literal_never_displaces_while_a_middle_tier_has_room():
    # Literal rule: displacement only when the only tier with room is the slowest. Block 2 finds
    # REMOTE free, so it goes there; 3 then finds only DISK and may displace -- but outranks no one.
    r = oa.run_online(BELIEFS, IS_NEW, 1.0, ONE_EACH, CM, EVENTS, new_ahead=False, policy="literal")
    # 2 REMOTE 1.0, 0 GPU .42, 3 DISK 1.6, 1 CPU .6, 2 REMOTE 1.0, 4 DISK 1.6, 3 DISK 1.6
    assert r["access_cost"] == pytest.approx(7.82)
    assert r["n_displacements"] == 0


# Cascade vs one-step: a new block displacing from GPU while CPU is full.
#   pre-split 0 (belief 3) GPU, 1 (belief 2) CPU; new 2 (belief 5); accesses 2 0 0 1.
#   onestep: 2 -> GPU, 0 -> REMOTE (fastest slower tier with room).  .42 + 1.0 + 1.0 + .6 = 3.02
#   cascade: 2 -> GPU, 0 -> CPU displacing 1 -> REMOTE.             .42 + .6 + .6 + 1.0 = 2.62
#   literal: REMOTE has room, no displacement: 2 -> REMOTE.          1.0 + .42 + .42 + .6 = 2.44
C_BEL, C_NEW, C_EV = [3.0, 2.0, 5.0], [False, False, True], [2, 0, 0, 1]


def test_cascade_differs_from_onestep_as_computed_by_hand():
    one = oa.run_online(C_BEL, C_NEW, 1.0, ONE_EACH, CM, C_EV, False, "onestep")
    cas = oa.run_online(C_BEL, C_NEW, 1.0, ONE_EACH, CM, C_EV, False, "cascade")
    lit = oa.run_online(C_BEL, C_NEW, 1.0, ONE_EACH, CM, C_EV, False, "literal")
    assert one["access_cost"] == pytest.approx(3.02) and one["moves"] == Counter({(GPU, REM): 1})
    assert cas["access_cost"] == pytest.approx(2.62)
    assert cas["moves"] == Counter({(GPU, CPU): 1, (CPU, REM): 1})
    assert lit["access_cost"] == pytest.approx(2.44) and lit["n_displacements"] == 0


def test_costaware_gate():
    mig = lambda i, j: oa.migration_cost(CM["tiers"][i], CM["tiers"][j], CM["read_ms"], W,  # noqa: E731
                                         "read+write")
    # belief 5 vs GPU resident 3: saving (5-3)*(1.0-0.42) = 1.16 > GPU->REMOTE 0.62 -> displaces.
    r = oa.run_online(C_BEL, C_NEW, 1.0, ONE_EACH, CM, C_EV, False, "costaware", mig_fn=mig)
    assert r["n_displacements"] == 1 and r["moves"] == Counter({(GPU, REM): 1})
    # belief 3.5: GPU saving 0.5*0.58 = 0.29 < 0.62; CPU saving (3.5-2)*(1.0-0.6) = 0.6 < 0.8.
    r = oa.run_online([3.0, 2.0, 3.5], C_NEW, 1.0, ONE_EACH, CM, C_EV, False, "costaware",
                      mig_fn=mig)
    assert r["n_displacements"] == 0 and r["where"] == [GPU, CPU, REM]


def test_new_blocks_at_belief_zero_never_displace():
    r = oa.run_online([3.0, 1.0, 0.0, 0.0, 0.0], IS_NEW, 1.0, ONE_EACH, CM, EVENTS, True, "cascade")
    assert r["n_displacements"] == 0 and r["where"] == [GPU, CPU, REM, DSK, DSK]


# --- the static per-access bill is place_with_beliefs, tier-only --------------------------------

TRACE = ROOT / "data" / "mooncake_conversation_trace.jsonl"


@pytest.fixture
def tier_only_sequential(monkeypatch):
    torch = pytest.importorskip("torch")
    import stoa.sequential as seq
    acts = oa.tier_only_actions()
    tiers = list(Tier)
    monkeypatch.setattr(seq, "LEGAL_ACTIONS", acts)
    monkeypatch.setattr(seq, "_ACTION_TIER", torch.tensor([tiers.index(a.tier) for a in acts]))
    return seq


@pytest.mark.skipif(not TRACE.exists(), reason="Mooncake trace not downloaded")
def test_static_per_access_bill_equals_place_with_beliefs(tier_only_sequential):
    seq = tier_only_sequential
    from stoa.mooncake import load_mooncake
    n_req = 400
    wl = load_mooncake(str(TRACE), max_requests=n_req)
    st = max(1, wl.time_split(0.5))
    pre, fut = wl.prefix_stats(st), wl.future_counts(st)
    ids = [i.item_id for i in wl.items]
    index = {b: k for k, b in enumerate(ids)}
    sizes = [float(i.size_bytes) for i in wl.items]
    caps = SIM.tier_capacities(wl.total_bytes)
    events = oa.post_split_events(str(TRACE), st, n_req, index)
    assert Counter(e for e in events if e >= 0) == Counter({index[b]: c for b, c in fut.items() if c > 0})
    counts = [float(pre[b]["count"]) for b in ids]
    is_new = [c == 0 for c in counts]
    assert any(is_new) and not all(is_new)

    def mine(bl, ahead=False):
        return oa.bill_static(events, oa.static_tiers(oa.rank_order(bl, is_new, ahead), sizes,
                                                      caps, CM["tiers"]), CM)

    heur = {b: counts[k] for k, b in enumerate(ids)}
    orac = {b: float(fut.get(b, 0)) for b in ids}
    for belief in (heur, orac):
        want = seq.place_with_beliefs(wl, belief)
        assert math.isclose(mine([belief[b] for b in ids]), want, rel_tol=1e-9)
    # prefix_greedy_cost accumulates float32 tensor costs, hence the looser tolerance.
    assert math.isclose(mine([heur[b] for b in ids]), seq.prefix_greedy_cost(wl), rel_tol=1e-6)
    # A tie-heavy arm (every new block at 1.0) in both conventions; new-ahead is the script's
    # +1e-9 nudge on the static side and a rank rule on ours.
    for ahead in (False, True):
        b = dict(heur)
        b.update({x: 1.0 + (1e-9 if ahead else 0.0) for k, x in enumerate(ids) if is_new[k]})
        bl = [1.0 if is_new[k] else counts[k] for k in range(len(ids))]
        assert math.isclose(mine(bl, ahead), seq.place_with_beliefs(wl, b), rel_tol=1e-9)
        # Cascade ends in exactly the static assignment, and never bills more (zero migration).
        r = oa.run_online(bl, is_new, sizes[0], caps, CM, events, ahead, "cascade")
        static = oa.static_tiers(oa.rank_order(bl, is_new, ahead), sizes, caps, CM["tiers"])
        assert [CM["tiers"][k] for k in r["where"]] == static
        assert r["access_cost"] <= mine(bl, ahead) + 1e-9
    # The online controller with new blocks at 0 is the heuristic, exactly.
    r = oa.run_online(counts, is_new, sizes[0], caps, CM, events, True, "onestep")
    assert r["n_displacements"] == 0
    assert math.isclose(r["access_cost"], seq.place_with_beliefs(wl, heur), rel_tol=1e-9)


def test_script_feeds_no_belief_from_the_future():
    tc = _load(ROOT / "tests" / "test_causal.py", "_test_causal_for_online")
    bad = tc.belief_taint(SCRIPT.read_text(), ("fut", "future_counts", "_truth"))
    assert not bad, bad
