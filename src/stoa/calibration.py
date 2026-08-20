"""Derive the tier-latency table instead of asserting it.

`simulator._TIER_READ_MS` began as four numbers chosen to look plausible. Scaling all four
by a common factor -- which is what `scripts/run_sensitivity.py` does -- leaves their
*ratios* untouched, and the ratios are what a placement policy actually responds to: an
item moves from CPU to GPU because GPU is 10x faster, not because it is 0.2 ms fast. So the
existing sensitivity analysis never tested the parameter that matters. This module exists
to fix that, and it is the same free-parameter defect the paper catalogs, found in our own
cost model.

The derivation is

    read_ms(tier) = overhead_ms(tier) + kv_bytes_per_block / bandwidth(tier)

with

    kv_bytes_per_block = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes x tokens

The second line is exact arithmetic over a model's published configuration: given the
architecture, there is nothing to estimate. The first line's `bandwidth` and `overhead` are
**declared assumptions about a device class, not measurements we took** -- this project has
no GPU and calibrating against a live vLLM+LMCache deployment remains open (see
docs/claims_dependency.md B). They are named, defaulted, and swept rather than buried.

The intended use is therefore NOT "these are the right numbers". It is: state the model and
device class explicitly, derive the table from them, and report the conclusion as a range
over the plausible space. `scripts/run_calibration_sensitivity.py` does that.

Profiles below are illustrative starting points. Verify `layers`/`kv_heads`/`head_dim`
against the model's own config before citing any single derived value.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environment import Tier


@dataclass(frozen=True)
class ModelProfile:
    """Attention shape needed to size a KV block. `source` names where to check it."""
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2          # fp16/bf16
    source: str = ""

    def kv_bytes_per_token(self) -> int:
        """Exact: 2 tensors (K, V) x layers x kv_heads x head_dim x bytes."""
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    def kv_bytes_per_block(self, block_tokens: int) -> int:
        return self.kv_bytes_per_token() * block_tokens


@dataclass(frozen=True)
class DeviceProfile:
    """Per-tier effective bandwidth (GB/s) and fixed overhead (ms).

    These are ASSUMPTIONS about a class of machine, not measurements. `bw_*` should be the
    bandwidth actually achieved on a bulk transfer, not a datasheet peak; `oh_*` absorbs
    everything not proportional to size (queueing, syscall, network round trip).
    """
    name: str
    bw_gpu: float
    bw_cpu: float
    bw_disk: float
    bw_remote: float
    oh_gpu: float = 0.0
    oh_cpu: float = 0.01
    oh_disk: float = 0.1
    oh_remote: float = 0.05
    note: str = ""

    def bandwidths(self) -> dict[Tier, float]:
        return {Tier.GPU: self.bw_gpu, Tier.CPU: self.bw_cpu,
                Tier.DISK: self.bw_disk, Tier.REMOTE_RDMA: self.bw_remote}

    def overheads(self) -> dict[Tier, float]:
        return {Tier.GPU: self.oh_gpu, Tier.CPU: self.oh_cpu,
                Tier.DISK: self.oh_disk, Tier.REMOTE_RDMA: self.oh_remote}


# --- Illustrative model profiles. Check these against the model's own config. -------------
MODELS: dict[str, ModelProfile] = {
    # GQA with 8 KV heads keeps the KV cache small relative to the parameter count; this is
    # the shape that makes KV tiering interesting at all.
    "llama3-8b-like": ModelProfile("llama3-8b-like", layers=32, kv_heads=8, head_dim=128,
                                   source="declared; verify against the model config"),
    "llama3-70b-like": ModelProfile("llama3-70b-like", layers=80, kv_heads=8, head_dim=128,
                                    source="declared; verify against the model config"),
    # A multi-head (non-GQA) shape, included because it moves bytes/token by ~8x and is the
    # honest upper end of the range rather than a specific released model.
    "mha-wide": ModelProfile("mha-wide", layers=32, kv_heads=32, head_dim=128,
                             source="synthetic upper bound, not a released model"),
}

# --- Illustrative device classes. The POINT is that these disagree about the ratios. -------
DEVICES: dict[str, DeviceProfile] = {
    "balanced": DeviceProfile(
        "balanced", bw_gpu=2000.0, bw_cpu=25.0, bw_disk=6.0, bw_remote=20.0,
        note="fast HBM, PCIe-class host link, NVMe, high-speed fabric"),
    "slow-fabric": DeviceProfile(
        "slow-fabric", bw_gpu=2000.0, bw_cpu=25.0, bw_disk=6.0, bw_remote=3.0,
        oh_remote=0.5,
        note="remote tier over a congested or commodity network: reorders remote vs disk"),
    "fast-storage": DeviceProfile(
        "fast-storage", bw_gpu=2000.0, bw_cpu=25.0, bw_disk=14.0, bw_remote=20.0,
        oh_disk=0.02,
        note="high-end local NVMe: compresses the disk-to-remote gap"),
    "narrow-host": DeviceProfile(
        "narrow-host", bw_gpu=2000.0, bw_cpu=8.0, bw_disk=6.0, bw_remote=20.0,
        note="constrained host link: CPU tier stops being the obvious second choice"),
    "flat": DeviceProfile(
        "flat", bw_gpu=200.0, bw_cpu=100.0, bw_disk=50.0, bw_remote=80.0,
        oh_cpu=0.0, oh_disk=0.0, oh_remote=0.0,
        note="deliberately compressed hierarchy: the control for 'do the ratios matter?'"),
}


def derive_tier_read_ms(model: ModelProfile, device: DeviceProfile,
                        block_tokens: int = 256) -> dict[Tier, float]:
    """Per-tier read latency (ms) for one KV block, from the model and device profiles.

    Mooncake's released traces use 256-token blocks and kv-cache-tester's use 64, which is
    why `block_tokens` is a parameter and not a constant: block size changes every tier's
    latency by the same factor, so it moves the absolute scale but not the ratios.
    """
    nbytes = model.kv_bytes_per_block(block_tokens)
    bw, oh = device.bandwidths(), device.overheads()
    # bytes / (GB/s) -> seconds; x1000 -> ms.
    return {t: oh[t] + (nbytes / (bw[t] * 1e9)) * 1e3 for t in Tier}


def tier_ratios(table: dict[Tier, float]) -> dict[str, float]:
    """Each tier's latency relative to the fastest, which is what a policy responds to."""
    fastest = min(table.values())
    return {t.value: round(v / fastest, 2) for t, v in table.items()}
