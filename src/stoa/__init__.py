"""STOA — Storage & Tiered Orchestration for Agents.

A learned memory-tier orchestrator for AI agents: jointly decides, per memory
item and under a latency+cost+token budget, its representation x storage tier x
update timing.

See docs/stoa_design.md for the full formulation.
"""
from .environment import (
    Action,
    Budget,
    ItemState,
    Representation,
    RewardWeights,
    Tier,
    Timing,
    reward,
)
from .orchestrator import Orchestrator
from .policy import OrchestratorPolicy, PolicyConfig
from .simulator import TieringSimulator
from .learn import ReusePredictor, fit_reuse_predictor
from .traces import Workload, WorkloadConfig, generate_workload

__version__ = "0.0.1"

__all__ = [
    "Action",
    "Budget",
    "ItemState",
    "Representation",
    "RewardWeights",
    "Tier",
    "Timing",
    "reward",
    "Orchestrator",
    "OrchestratorPolicy",
    "PolicyConfig",
    "TieringSimulator",
    "Workload",
    "WorkloadConfig",
    "generate_workload",
    "ReusePredictor",
    "fit_reuse_predictor",
]
