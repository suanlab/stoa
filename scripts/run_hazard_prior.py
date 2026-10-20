#!/usr/bin/env python3
"""H1: replace the tie CONVENTION with an ESTIMATE of how much to trust an unseen block.

At arrival a new block and a pre-split block seen once both have count 1, so a count-based belief
ties them and a convention decides (it moved the order arms 14-29 points; notebook AR, AS). They are
not alike: the new block has its whole reuse life ahead, the old one has survived unreused. Here the
relative belief is estimated from the PREFIX only, at a pseudo-split p = split / 2:
  * old-at-p blocks: label = accesses in [p, split), binned by count at p and age at p;
  * blocks born in [p, split) -- the analogue of post-split arrivals: label = their accesses up to split.
Labels are rescaled by (end - split) / (split - p) so magnitudes match the evaluation window.
Stated conventions (unswept unless noted): one pseudo-split (split / 2); count bins (1,2,3,5,9), each
represented by its lower edge; age quartiles within a bin; c* nudged off integers so it never ties; every
new block gets the same c*, so the stable placement sort admits new blocks in arrival order.

Arms (prefix-heuristic beliefs for old blocks unless stated; both tie conventions; --tier-only patches
the action space as run_tier_only.py does):
  hazard-new   new blocks at the point c* of the count scale where an old block's expected reuse
               equals a new block's (linear interpolation between integer counts: never a tie)
  hazard-all   every block at its estimated expected reuse: old by (count, age) bin, new by the
               new-block mean
  hazard-old   old blocks by (count, age), new blocks left at 0 -- isolates the old-block part
H1 predicts: behind/ahead differ by < 2 points for hazard-new, and it is within 2 points of the better
convention's constant arms in every configuration (falsified if > 5 or below the worse convention).

Usage: python3 scripts/run_hazard_prior.py [--tier-only]
"""
from __future__ import annotations

import sys

TIER_ONLY = "--tier-only" in sys.argv
# Sensitivity (review of 2026-10-20): count bins are estimated over [0, p) but applied to counts over
# [0, split), twice the exposure. --rescale-cstar multiplies c* by split / p as the crude alignment.
RESCALE = "--rescale-cstar" in sys.argv
if TIER_ONLY:                                             # before any placement module is imported
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))
    from stoa.environment import Action, Representation, Timing
    _T = [a for a in Action.space() if a.representation == Representation.PLAINTEXT
          and a.timing == Timing.ONLINE]
    Action.space = staticmethod(lambda: list(_T))

import bisect  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_arrival_admission import TRACES  # noqa: E402
from stoa.mooncake import load_mooncake  # noqa: E402

from stoa.sequential import place_with_beliefs, prefix_greedy_cost  # noqa: E402

EPS = 1e-9
COUNT_BINS = (1, 2, 3, 5, 9, 10 ** 9)          # [1,2) [2,3) [3,5) [5,9) [9,inf)


def cbin(c):
    return bisect.bisect_right(COUNT_BINS, c) - 1


def estimate_prior(wl, rescale: bool = False) -> dict:
    """The prefix-only estimate of how far to trust an unseen block (pseudo-split at split / 2).

    Returns c* (where new blocks sit on the old blocks' count scale), the new-block and per-count
    expected reuse, and `hz_old(b)` for old blocks. Shared with run_online_admission.py.
    """
    st = max(1, wl.time_split(0.5))
    end = wl.accesses[-1][0] + 1
    p = st // 2
    scale = (end - st) / (st - p)
    steps = {b: tr.access_steps for b, tr in wl.traces.items()}
    # --- estimation, prefix only -------------------------------------------------------
    old_lab, old_age = defaultdict(list), {}
    new_lab = []
    for b, s in steps.items():
        first = s[0]
        if first >= st:
            continue
        if first < p:
            c = bisect.bisect_left(s, p)
            old_lab[cbin(c)].append((p - first, bisect.bisect_left(s, st) - c))
        else:
            # its accesses up to the split, the creation access excluded: a compulsory miss,
            # exactly as Workload.future_counts bills new blocks after the split
            new_lab.append(bisect.bisect_left(s, st) - 1)
    h_new = scale * sum(new_lab) / len(new_lab)
    # age quartiles within each count bin, from the estimation sample
    table, m_count = {}, {}
    for cb, pairs in old_lab.items():
        ages = sorted(a for a, _ in pairs)
        cuts = [ages[len(ages) * k // 4] for k in (1, 2, 3)]
        buckets = defaultdict(list)
        for a, lab in pairs:
            buckets[bisect.bisect_right(cuts, a)].append(lab)
        table[cb] = (cuts, {k: scale * sum(v) / len(v) for k, v in buckets.items()})
        m_count[cb] = scale * sum(l for _, l in pairs) / len(pairs)
    # c*: where the old blocks' expected reuse by count crosses the new-block mean
    reps = [1, 2, 3, 5, 9]                         # representative count of each bin
    # A bin with no estimation blocks (synthetic has none at some counts) borrows the nearest
    # lower populated bin, so the scale stays monotone in count.
    for k in range(len(reps)):
        if k not in table:
            lower = max((j for j in table if j < k), default=min(table))
            table[k], m_count[k] = table[lower], m_count[lower]
    ms = [m_count[k] for k in range(len(reps))]
    if h_new <= ms[0]:
        c_star = max(EPS, h_new / ms[0]) if ms[0] > 0 else 0.5   # between 0 and 1
    else:
        c_star = reps[-1] + 1.0
        for k in range(len(reps) - 1):
            if ms[k] < h_new <= ms[k + 1]:
                c_star = reps[k] + (h_new - ms[k]) / (ms[k + 1] - ms[k]) * (reps[k + 1] - reps[k])
                break
    if rescale:
        c_star *= st / p
    if abs(c_star - round(c_star)) < 1e-6:
        c_star += 1e-3                              # never land exactly on an integer count
    est = {"pseudo_split": p, "scale": round(scale, 3), "h_new": round(h_new, 4),
                       "m_by_count_bin": [round(v, 4) for v in ms], "c_star": round(c_star, 4)}
    steps = {b: tr.access_steps for b, tr in wl.traces.items()}
    pre = wl.prefix_stats(st)

    def hz_old(b):
        c = int(pre[b]["count"])
        cuts, vals = table[cbin(c)]
        a = st - steps[b][0]
        return vals.get(bisect.bisect_right(cuts, a), m_count[cbin(c)])

    return {"st": st, "c_star": c_star, "h_new": h_new, "estimates": est, "hz_old": hz_old}


def main() -> int:
    out_rows, estimates = [], {}
    for name, path in TRACES.items():
        wl = load_mooncake(str(ROOT / path))
        E = estimate_prior(wl, RESCALE)
        st, c_star, hz_old, h_new = E["st"], E["c_star"], E["hz_old"], E["h_new"]
        estimates[name] = E["estimates"]
        steps = {b: tr.access_steps for b, tr in wl.traces.items()}
        # --- evaluation at the real split ---------------------------------------------------
        pre = wl.prefix_stats(st)
        fut = wl.future_counts(st)
        base = {i.item_id: float(pre[i.item_id]["count"]) for i in wl.items}
        new_ids = [b for b, s in steps.items() if s[0] >= st]
        old_ids = [b for b in base if steps[b][0] < st]

        c_pref = prefix_greedy_cost(wl)
        c_orac = place_with_beliefs(wl, {b: float(v) for b, v in fut.items()})
        span = c_pref - c_orac
        arms = {
            "hazard-new": ({}, {b: c_star for b in new_ids}),
            "hazard-all": ({b: hz_old(b) for b in old_ids}, {b: h_new for b in new_ids}),
            "hazard-old": ({b: hz_old(b) for b in old_ids}, {}),
        }
        for tie in ("new-behind", "new-ahead"):
            for arm, (old_over, new_bel) in arms.items():
                bel = dict(base)
                bel.update(old_over)
                eps = EPS if tie == "new-ahead" else 0.0
                bel.update({b: v + eps for b, v in new_bel.items()})
                g = 100.0 * (c_pref - place_with_beliefs(wl, bel)) / span
                out_rows.append({"trace": name, "actions": "tier" if TIER_ONLY else "all", "tie": tie,
                                 "arm": arm, "captured_pct": round(g, 2)})
                print(f"{name:<13} {tie:<10} {arm:<12} {g:7.2f}%")
        print(name, estimates[name])
    dest = ROOT / "experiments" / f"hazard_prior{'_tier' if TIER_ONLY else ''}{'_rescaled' if RESCALE else ''}.json"
    if "--out" in sys.argv:                       # used by run_bailian.py's workers
        dest = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    dest.write_text(json.dumps({"config": {"pseudo_split": "split // 2", "count_bins": list(COUNT_BINS[:-1]),
                                           "age": "quartiles within count bin",
                                           "actions": "tier" if TIER_ONLY else "all"},
                                "estimates": estimates, "rows": out_rows}, indent=2) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
