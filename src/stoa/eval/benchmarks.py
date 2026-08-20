"""Benchmark registry (docs/stoa_design.md §5).

STOA is evaluated on long-term agent-memory benchmarks, with retrieval
components stress-tested on the Big ANN streaming/filtered tracks. Metrics:
accuracy x latency x $/token x freshness -> Pareto hypervolume, with a Belady
oracle upper bound.

Scaffold: specs + loader stubs. Wire real dataset adapters in M3-6.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    name: str
    arxiv: str
    probes: str          # what memory ability it stresses
    metric: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "longmemeval": BenchmarkSpec(
        key="longmemeval",
        name="LongMemEval",
        arxiv="2410.10813",
        probes="extraction, multi-session/temporal reasoning, knowledge updates, abstention",
        metric="QA accuracy",
    ),
    "locomo": BenchmarkSpec(
        key="locomo",
        name="LoCoMo",
        arxiv="2402.17753",
        probes="~300-turn multi-session conversational memory",
        metric="QA accuracy",
    ),
    "memoryagentbench": BenchmarkSpec(
        key="memoryagentbench",
        name="MemoryAgentBench",
        arxiv="2507.05257",
        probes="retrieval, test-time learning, long-range understanding, conflict resolution",
        metric="per-competency accuracy",
    ),
    "bigann": BenchmarkSpec(
        key="bigann",
        name="Big ANN (NeurIPS'23)",
        arxiv="2409.17424",
        probes="filtered / streaming ANN index stress test",
        metric="Recall@k vs QPS",
    ),
}


def load(benchmark_key: str):
    """Return an iterable of budget-QA tasks for `benchmark_key`.

    LoCoMo is implemented (eval/locomo.py). The others still raise until their
    dataset adapters land — honest stubs, not fake data.
    """
    if benchmark_key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark: {benchmark_key!r}; choose from {list(BENCHMARKS)}")
    if benchmark_key == "locomo":
        from .locomo import load_locomo
        return load_locomo()
    raise NotImplementedError(
        f"{BENCHMARKS[benchmark_key].name} adapter not yet implemented (roadmap M3-6)."
    )
