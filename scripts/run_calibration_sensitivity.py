#!/usr/bin/env python3
"""Does the headroom claim survive a cost model derived from hardware rather than invented?

`scripts/run_sensitivity.py` multiplies every tier latency by a common factor. That moves
the absolute scale and leaves the RATIOS between tiers untouched -- and the ratios are what
a placement policy responds to. So it never tested the parameter that matters. This script
does, by deriving the whole table from a model's attention shape and a device class
(`stoa.calibration`) and sweeping both.

Two things come out of it. First, the hand-written table this project has used puts GPU:CPU
at 1:10, while a bandwidth derivation puts it near 1:80 -- an order of magnitude, in the
ratio the policy is most sensitive to. Second, some device classes REORDER the hierarchy:
on a constrained host link the CPU tier becomes slower than the remote one, which is the
non-monotonicity that `tests/test_rl.py::test_tier_hierarchy_is_monotonic` exists to catch.
Both are reported rather than smoothed over.

This is still not a calibration against a live deployment -- there is no GPU here and that
gate remains open (docs/claims_dependency.md B). It is the honest intermediate: an explicit
derivation from declared inputs, swept, with the conclusion reported as a range.

CPU, about a minute. Usage: python3 scripts/run_calibration_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import stoa.simulator as sim_mod  # noqa: E402
from stoa.calibration import DEVICES, MODELS, derive_tier_read_ms, tier_ratios  # noqa: E402
from stoa.environment import Tier  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402
from stoa.sequential import place_with_beliefs, reference_costs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_session_features import build_features, fit_logistic, predict  # noqa: E402


def _monotonic(table: dict[Tier, float]) -> bool:
    """Latency must rise as capacity does, or a cost-minimizing policy skips a whole tier.

    This is the invariant an earlier version of the simulator violated: remote was both
    faster AND larger than disk, so nothing would ever choose disk and the four-tier
    hierarchy was really three.
    """
    cap = sim_mod._TIER_CAPACITY_FRACTION
    order = sorted(Tier, key=lambda t: table[t])
    caps = [cap[t] for t in order]
    return all(a <= b for a, b in zip(caps, caps[1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Hardware-derived cost-model sensitivity")
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--block-tokens", type=int, default=256,
                    help="Mooncake blocks are 256 tokens; kv-cache-tester's are 64")
    ap.add_argument("--out", type=str, default="experiments/calibration_sensitivity.json")
    args = ap.parse_args()

    paths = {"toolagent": "data/mooncake_toolagent_trace.jsonl",
             "conversation": "data/mooncake_conversation_trace.jsonl"}
    for pth in paths.values():
        if not Path(pth).exists():
            sys.exit(f"missing {pth} — see src/stoa/mooncake.py for the download command")
    n_req = args.requests or 10 ** 9
    wl = load_mooncake(paths["toolagent"], max_requests=args.requests or None)

    # Features and the fitted model do NOT depend on the cost table -- only the billing does.
    # Building them once and reusing across all 15 cost models is what makes this sweep
    # affordable at full trace length, and it also guarantees that the ONLY thing varying
    # across rows is the cost model.
    Xe, _, ye, ids = build_features(paths["toolagent"], n_req)
    Xt, _, yt, _ = build_features(paths["conversation"], n_req)
    scores = predict(fit_logistic(Xt, yt), Xe)
    lo, hi = min(scores), max(scores)
    norm = [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in scores]
    st = max(1, wl.time_split(0.5))
    fut = wl.future_counts(st)
    reused = [v for v in fut.values() if v > 0]
    mean_reuse = sum(reused) / max(1, len(reused))
    belief = {i: p * mean_reuse for i, p in zip(ids, norm)}

    def captured() -> float:
        """Fraction of the oracle interval the learned belief recovers, under whatever cost
        table is currently installed. This is the paper's headline quantity, and until now it
        was the one thing the cost-model sweep did not vary."""
        r = reference_costs(wl)
        span = r["prefix_greedy"] - r["oracle_future"]
        if span <= 0:
            return 0.0
        c = place_with_beliefs(wl, belief)
        return 100.0 * (r["prefix_greedy"] - c) / span

    base = dict(sim_mod._TIER_READ_MS)
    rows = []
    print(f"\nHardware-derived cost-model sweep (toolagent, {args.requests} requests, "
          f"{args.block_tokens}-token blocks)")
    print(f"{'model':<16} {'device':<13} | {'GPU:CPU':>8} {'GPU:disk':>9} | "
          f"{'headroom':>8} | {'captured':>8} | mono")
    print("-" * 80)

    # The table this project has been using, for reference.
    r0 = reference_costs(wl)
    cap0 = captured()
    print(f"{'(hand-written)':<16} {'—':<13} | "
          f"{tier_ratios(base)['cpu']:8.1f} {tier_ratios(base)['disk']:9.1f} | "
          f"{r0['headroom_pct']:7.1f}% | {cap0:+7.1f}% | {'yes' if _monotonic(base) else 'NO'}")
    rows.append({"model": "hand-written", "device": None,
                 "ratios": tier_ratios(base), "headroom_pct": round(r0["headroom_pct"], 2),
                 "captured_pct": round(cap0, 2), "monotonic": _monotonic(base)})

    try:
        for mname, model in MODELS.items():
            for dname, device in DEVICES.items():
                table = derive_tier_read_ms(model, device, args.block_tokens)
                sim_mod._TIER_READ_MS.update(table)
                r = reference_costs(wl)
                cap = captured()
                ratios = tier_ratios(table)
                mono = _monotonic(table)
                print(f"{mname:<16} {dname:<13} | {ratios['cpu']:8.1f} {ratios['disk']:9.1f} | "
                      f"{r['headroom_pct']:7.1f}% | {cap:+7.1f}% | {'yes' if mono else 'NO'}",
                      flush=True)
                rows.append({"model": mname, "device": dname,
                             "kv_bytes_per_token": model.kv_bytes_per_token(),
                             "read_ms": {t.value: round(v, 4) for t, v in table.items()},
                             "ratios": ratios,
                             "headroom_pct": round(r["headroom_pct"], 2),
                             "captured_pct": round(cap, 2), "monotonic": mono})
    finally:
        sim_mod._TIER_READ_MS.update(base)          # never leave the module perturbed

    derived = [r for r in rows if r["device"]]
    hs = [r["headroom_pct"] for r in derived]
    cs = [r["captured_pct"] for r in derived]
    print(f"\ncaptured headroom across {len(derived)} derived cost models: "
          f"{min(cs):+.1f}%--{max(cs):+.1f}% (hand-written table: {cap0:+.1f}%)")
    nonmono = [f"{r['model']}/{r['device']}" for r in derived if not r["monotonic"]]
    print(f"\nheadroom across {len(derived)} derived cost models: "
          f"{min(hs):.1f}%--{max(hs):.1f}% (hand-written table: {r0['headroom_pct']:.1f}%)")
    if nonmono:
        print(f"NON-MONOTONIC hierarchies (latency order disagrees with capacity order): "
              f"{', '.join(nonmono)}")
        print("  These are not bugs in the derivation -- they are device classes where the")
        print("  tier a policy should prefer genuinely changes. A fixed capacity table")
        print("  cannot describe them, which is a limitation of the simulator, not of them.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"requests": args.requests, "block_tokens": args.block_tokens,
                    "note": "derived from declared model/device profiles, NOT measured"},
         "summary": {"headroom_min": round(min(hs), 2), "headroom_max": round(max(hs), 2),
                     "captured_min": round(min(cs), 2), "captured_max": round(max(cs), 2),
                     "handwritten_captured": round(cap0, 2),
                     "handwritten_headroom": round(r0["headroom_pct"], 2),
                     "n_models": len(MODELS), "n_devices": len(DEVICES),
                     "non_monotonic": nonmono},
         "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
