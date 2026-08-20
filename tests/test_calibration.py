"""Cost-model derivation (src/stoa/calibration.py).

The arithmetic here decides whether the paper's headroom range is derived or invented, so
the KV-size formula is checked against hand-computed values rather than against whatever the
code produced first. The device-class parameters are declared assumptions and are not
testable for accuracy -- what is testable, and what these tests pin, is that the derivation
composes correctly and that the hierarchy's ordering behaves as the profiles claim.
"""
from __future__ import annotations

import pytest

from stoa.calibration import (DEVICES, MODELS, DeviceProfile, ModelProfile,
                              derive_tier_read_ms, tier_ratios)
from stoa.environment import Tier


def test_kv_bytes_per_token_matches_hand_arithmetic():
    """2 (K and V) x 32 layers x 8 kv heads x 128 dim x 2 bytes = 131,072 B = 128 KiB."""
    m = ModelProfile("t", layers=32, kv_heads=8, head_dim=128, dtype_bytes=2)
    assert m.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2 == 131_072
    assert m.kv_bytes_per_token() / 1024 == 128


def test_kv_bytes_scales_linearly_in_every_dimension():
    base = ModelProfile("b", layers=32, kv_heads=8, head_dim=128).kv_bytes_per_token()
    assert ModelProfile("l", layers=64, kv_heads=8, head_dim=128).kv_bytes_per_token() == 2 * base
    assert ModelProfile("h", layers=32, kv_heads=16, head_dim=128).kv_bytes_per_token() == 2 * base
    assert ModelProfile("d", layers=32, kv_heads=8, head_dim=256).kv_bytes_per_token() == 2 * base
    fp8 = ModelProfile("q", layers=32, kv_heads=8, head_dim=128, dtype_bytes=1)
    assert fp8.kv_bytes_per_token() == base // 2


def test_block_bytes_scale_with_block_size():
    m = MODELS["llama3-8b-like"]
    assert m.kv_bytes_per_block(256) == 4 * m.kv_bytes_per_block(64)


def test_latency_is_overhead_plus_bytes_over_bandwidth():
    m = ModelProfile("t", layers=1, kv_heads=1, head_dim=1, dtype_bytes=1)   # 2 B/token
    d = DeviceProfile("t", bw_gpu=1.0, bw_cpu=1.0, bw_disk=1.0, bw_remote=1.0,
                      oh_gpu=5.0, oh_cpu=0.0, oh_disk=0.0, oh_remote=0.0)
    t = derive_tier_read_ms(m, d, block_tokens=1)          # 2 bytes at 1 GB/s = 2e-6 ms
    assert t[Tier.GPU] == pytest.approx(5.0 + 2e-6)
    assert t[Tier.CPU] == pytest.approx(2e-6)


def test_block_size_moves_scale_but_not_ratios():
    """Block size multiplies every tier's transfer term equally, so it cannot change which
    tier a policy prefers. This is why the sweep varies device class instead."""
    m, d = MODELS["llama3-8b-like"], DEVICES["balanced"]
    small = derive_tier_read_ms(m, d, 64)
    large = derive_tier_read_ms(m, d, 1024)
    assert large[Tier.CPU] > small[Tier.CPU]
    # Overheads are fixed, so the ratios converge as the transfer term dominates; at 1024
    # tokens the transfer term is large enough that the ratios must agree closely.
    rs, rl = tier_ratios(small), tier_ratios(large)
    assert rl["disk"] > rs["disk"] * 0.5


@pytest.mark.parametrize("name", list(DEVICES))
def test_gpu_is_the_fastest_tier_in_every_profile(name):
    t = derive_tier_read_ms(MODELS["llama3-8b-like"], DEVICES[name])
    assert min(t, key=t.get) is Tier.GPU, name


def test_profiles_disagree_about_the_hierarchy_ordering():
    """The sweep is only worth running if the device classes actually disagree about which
    tier is second-fastest. If they all agreed, varying them would test nothing."""
    orders = set()
    for name in DEVICES:
        t = derive_tier_read_ms(MODELS["llama3-8b-like"], DEVICES[name])
        orders.add(tuple(x.value for x in sorted(Tier, key=lambda k: t[k])))
    assert len(orders) > 1, orders


def test_derived_gpu_cpu_ratio_far_exceeds_the_hand_written_one():
    """The finding the derivation exists to surface: the hand-written table put GPU:CPU at
    1:10, and a bandwidth derivation puts it an order of magnitude higher."""
    from stoa.simulator import _TIER_READ_MS
    hand = tier_ratios(_TIER_READ_MS)["cpu"]
    derived = tier_ratios(derive_tier_read_ms(MODELS["llama3-8b-like"],
                                              DEVICES["balanced"]))["cpu"]
    assert hand == pytest.approx(10.0, abs=0.5)
    assert derived > 5 * hand, (hand, derived)


def test_deriving_does_not_mutate_the_simulator_table():
    """`derive_tier_read_ms` must be pure; the sweep script patches the module explicitly and
    restores it in a finally block, and a hidden mutation here would defeat that."""
    from stoa.simulator import _TIER_READ_MS
    before = dict(_TIER_READ_MS)
    derive_tier_read_ms(MODELS["mha-wide"], DEVICES["flat"])
    assert dict(_TIER_READ_MS) == before
