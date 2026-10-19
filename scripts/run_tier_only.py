#!/usr/bin/env python3
"""The split-instant ceiling and the arrival-time arms with placement restricted to the TIER.

`place_with_beliefs` chooses a full action -- representation x tier x timing -- by the simulator's
cost tables, and a belief of 0 versus >= 1 flips the representation (PLAINTEXT/ONLINE versus
LATENT/SLEEP). Those tables are placeholders, and the paper says the traces exercise the tier
axis only. This control runs the same two experiments with every action fixed to PLAINTEXT and
ONLINE, so the only decision left is the tier: for a KV block, the one choice that is physically
meaningful. Found on the 2026-10-19 second re-verification pass (notebook §AR).

It patches `Action.space` before any placement module is imported, so the baseline, the oracle,
the ceiling and every arm see the same restricted action set, then runs the unmodified producers.

CPU, about ten minutes. Usage: python3 scripts/run_tier_only.py
"""
from __future__ import annotations

import json
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stoa.environment import Action, Representation, Timing  # noqa: E402

# Read before any producer runs: _run() overwrites sys.argv.
REUSE = "--ref-scale" in sys.argv[1:] and sys.argv[sys.argv.index("--ref-scale") + 1] == "reuse"

_FULL = Action.space()
TIER_ONLY = [a for a in _FULL if a.representation == Representation.PLAINTEXT
             and a.timing == Timing.ONLINE]
Action.space = staticmethod(lambda: list(TIER_ONLY))


def _run(script: str, *args: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        sys.argv = [script, *args, "--out", str(out)]
        try:
            runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
        except SystemExit as e:              # the producers end in sys.exit(main())
            if e.code not in (0, None):
                raise
        return json.loads(out.read_text())


def main() -> int:
    import stoa.gnn  # noqa: F401  (imported after the patch, so LEGAL_ACTIONS is tier-only)
    assert len(stoa.gnn.LEGAL_ACTIONS) == len(TIER_ONLY), "the patch did not reach LEGAL_ACTIONS"
    reach = _run("run_reachable_ceiling.py", "--splits", "0.5")
    reuse = REUSE
    arrival = _run("run_arrival_admission.py", *(["--ref-scale", "reuse"] if reuse else []))
    dest = ROOT / "experiments" / ("tier_only_reuse.json" if reuse else "tier_only.json")
    dest.write_text(json.dumps({
        "config": {"actions": [a.tier.value for a in TIER_ONLY],
                   "representation": "plaintext", "timing": "online"},
        "reachable_ceiling": reach, "arrival_admission": arrival}, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
