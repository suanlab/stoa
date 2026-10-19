#!/usr/bin/env python3
"""Online admission control for the arrival-time arms, tier-only (H6).

`run_arrival_admission.py` forms a belief for every NEW block (first seen at or after the split)
causally, but `place_with_beliefs` then allocates tier capacity in ONE static greedy pass over all
blocks sorted by belief. A block that arrives late in the evaluation window is therefore holding a
fast tier from the split onward, ahead of pre-split blocks it outranks, without ever displacing
anyone. That is a bound on an admission point, not an admission point (notebook §AR.3).

This script runs the same beliefs through an ONLINE controller:

  * actions are restricted to PLAINTEXT/ONLINE exactly as `run_tier_only.py` does (only the tier
    is chosen);
  * at the split, pre-split blocks are placed by the frequency heuristic exactly as the baseline
    places them (heuristic counts, stable order, fastest tier with room);
  * post-split requests are replayed in time order, blocks in request order. A new block is
    admitted at its first appearance: it takes the fastest tier with room, unless a tier faster
    than that one holds a resident it outranks, in which case it displaces that tier's
    lowest-ranked resident, which is demoted (and pays a migration);
  * every post-split access is billed at the tier its block occupies at that moment.

The per-access bill is derived from `TieringSimulator.cost_of_placement` and validated against
`place_with_beliefs` for a static assignment before anything else is run.

CPU, a few minutes. Usage: python3 scripts/run_online_admission.py [--requests N]
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Only modules that do NOT import stoa.gnn may be imported here: the tier-only restriction has to
# be applied before stoa.gnn builds LEGAL_ACTIONS (see run_tier_only.py), and a test process that
# imports this module must not have its action space changed under it.
from stoa.credit import weighted_cost  # noqa: E402
from stoa.environment import Action, Representation, Tier, Timing  # noqa: E402
from stoa.eval.oracle import DEFAULT_WEIGHTS  # noqa: E402
from stoa.simulator import TieringSimulator  # noqa: E402

TIES = ("new-behind", "new-ahead")          # the names run_arrival_admission.py uses
POLICIES = ("onestep", "cascade", "literal")
ARMS = ("order-arrival", "order-learned", "order-shuffled", "constant-mean", "constant-median")


def tier_only_actions() -> list[Action]:
    """The action set run_tier_only.py restricts to: PLAINTEXT, ONLINE, any tier."""
    full = [Action(r, t, ts) for r in Representation for t in Tier for ts in Timing
            if Action(r, t, ts).is_legal()]
    return [a for a in full if a.representation == Representation.PLAINTEXT
            and a.timing == Timing.ONLINE]


# --- the cost model ------------------------------------------------------------------------------

def cost_model(sim: TieringSimulator, w, actions, rep: Representation) -> dict:
    """Per-access and fixed cost of each tier, DERIVED from `cost_of_placement`.

    `cost_of_placement(a, n)` is a one-off promotion cost plus `n` x (read latency + tokens), so
    weighted cost is affine in n. We read the intercept and slope off n = 0 and n = 1 and assert
    affinity at other n rather than assume it. Returns tiers ordered fast -> slow.
    """
    def c(a, n):
        return weighted_cost(sim.cost_of_placement(a, n, rep), w)

    per, fixed = {}, {}
    for a in actions:
        c0, c1 = c(a, 0), c(a, 1)
        for n in (2, 3, 17, 1000, 123457):
            cn = c(a, n)
            assert abs(cn - (c0 + n * (c1 - c0))) <= 1e-9 * max(1.0, abs(cn)), \
                f"cost_of_placement is not affine in n for {a}"
        per[a.tier], fixed[a.tier] = c1 - c0, c0
    tiers = sorted(per, key=sim.read_latency)
    # The admission rule below takes "the fastest tier with room". That equals the argmin
    # `place_with_beliefs` computes (min cost at the belief, ties by latency) only if the per-access
    # cost increases with latency and the fixed cost does not depend on the tier. Asserted.
    assert all(per[a] < per[b] for a, b in zip(tiers, tiers[1:])), "per-access cost not monotone"
    assert len(set(fixed.values())) == 1, "fixed cost depends on the tier"
    return {"tiers": tiers, "per": per, "fixed": fixed,
            "read_ms": {t: sim.read_latency(t) for t in tiers}}


def migration_cost(src: Tier, dst: Tier, read_ms: dict, w, model: str) -> float:
    """Cost of moving one block between tiers. The simulator has none (its only one-off cost is a
    REPRESENTATION change, `_PROMOTE_COST`, which is 0 for PLAINTEXT and is never incurred by a
    tier move), so this is an ASSUMPTION: one read from the source plus one write to the
    destination, the write taken to cost the destination's read latency, weighted by lambda_L.
    No token term: a migration does not serve the block into a context."""
    if model == "zero":
        return 0.0
    if model == "read+write":
        return w.lambda_latency * (read_ms[src] + read_ms[dst])
    raise ValueError(model)


# --- static assignment and per-access billing ------------------------------------------------------

def static_tiers(order: list[int], sizes: list[float], caps: dict, tiers: list) -> list:
    """Replicates `place_with_beliefs`'s capacity pass for tier-only actions: blocks in `order`
    (belief descending, stable), each to the fastest tier with room (`remaining < size` skips, as
    there). Returns tier per block index."""
    remaining = {t: caps[t] for t in tiers}
    out = [None] * len(sizes)
    for b in order:
        for t in tiers:
            if remaining[t] < sizes[b]:
                continue
            remaining[t] -= sizes[b]
            out[b] = t
            break
    return out


def rank_order(beliefs: list[float], is_new: list[bool], new_ahead: bool) -> list[int]:
    """The order `place_with_beliefs` visits blocks in: belief descending, then item order.

    Every pre-split block precedes every new block in item order, so at equal belief a new block
    is visited behind a pre-split one ("new-behind"); run_arrival_admission's "new-ahead" adds
    1e-9 to new beliefs, which is the same as ranking new ahead at exact ties."""
    def grp(b):
        return (0 if is_new[b] else 1) if new_ahead else (1 if is_new[b] else 0)
    return sorted(range(len(beliefs)), key=lambda b: (-beliefs[b], grp(b), b))


def bill_static(events: list[int], tier_of: list, cm: dict) -> float:
    """Every post-split access at the tier its block holds, plus each block's fixed cost once."""
    per, fixed = cm["per"], cm["fixed"]
    return (sum(per[tier_of[b]] for b in events if b >= 0)        # creation events cost nothing
            + sum(fixed[t] for t in tier_of if t is not None))


# --- the online controller ---------------------------------------------------------------------

def run_online(beliefs: list[float], is_new: list[bool], size: float, caps: dict, cm: dict,
               events: list[int], new_ahead: bool, policy: str, mig_fn=None) -> dict:
    """Replay post-split accesses with online admission. Returns access cost and moves.

    beliefs   per block; pre-split blocks carry their heuristic count (fixed), new blocks the arm's
    is_new    True for blocks with no pre-split access
    events    post-split accesses as block indices, in time order (request order within a step)
    new_ahead tie rule against a pre-split resident of EQUAL belief: True = the new block wins.
              Between two new blocks of equal belief the earlier (incumbent) always wins, exactly as
              the static pass's stable sort orders them in both conventions.
    policy    'onestep': displace into the fastest full tier whose lowest-ranked resident the new
                         block outranks; the victim goes to the fastest slower tier with room.
              'cascade': as onestep, but the victim is re-admitted below, displacing a resident it
                         outranks there in turn (each move is a migration).
              'literal': displacement considered only when the only tier with room is the slowest.
              'costaware': as onestep, but a displacement is made only if the BELIEVED saving pays
                         for the move: b_new*(c[F]-c[T]) - b_victim*(c[dest]-c[T]) > mig_fn(T, dest),
                         with c the per-access cost and F the tier the new block would otherwise get.
                         Uses beliefs only (causal). With mig_fn = 0 it reduces to onestep at
                         strict belief inequality.

    Block sizes must be uniform (true of Mooncake), so one demotion always frees one slot.
    """
    tiers = cm["tiers"]
    K = len(tiers)
    per = [cm["per"][t] for t in tiers]
    fixed = [cm["fixed"][t] for t in tiers]
    remaining = [caps[t] for t in tiers]
    n = len(beliefs)
    where = [-1] * n
    heaps: list[list] = [[] for _ in range(K)]

    def grp(b):
        return (0 if is_new[b] else 1) if new_ahead else (1 if is_new[b] else 0)

    def rk(b):                                   # smaller ranks ahead
        return (-beliefs[b], grp(b), b)

    def put(b, k):
        where[b] = k
        remaining[k] -= size
        heapq.heappush(heaps[k], (beliefs[b], -grp(b), -b))   # min-heap = lowest-ranked first

    def worst(k):
        h = heaps[k]
        while h:
            b = -h[0][2]
            if where[b] == k:                    # blocks only move down, so stale = moved away
                return b
            heapq.heappop(h)
        return None

    def take(b):
        remaining[where[b]] += size
        where[b] = -1

    fixed_cost = 0.0
    # At the split: pre-split blocks by the frequency heuristic, exactly as the baseline places
    # them (belief descending, stable, fastest tier with room).
    for b in sorted((b for b in range(n) if not is_new[b]), key=lambda b: (-beliefs[b], b)):
        k = next(k for k in range(K) if remaining[k] >= size)
        put(b, k)
        fixed_cost += fixed[k]

    moves: Counter = Counter()
    n_disp = 0
    admitted = []

    def admit(x):
        nonlocal n_disp
        F = next(k for k in range(K) if remaining[k] >= size)
        cands = range(F) if (policy != "literal" or F == K - 1) else range(0)
        for T in cands:
            v = worst(T)
            if v is None or not rk(v) > rk(x):
                continue
            if policy == "costaware":
                # tiers T+1..F-1 are full, so after the swap the victim's destination is F
                gain = beliefs[x] * (per[F] - per[T]) - beliefs[v] * (per[F] - per[T])
                if not gain > mig_fn(T, F):
                    continue
            take(v)
            put(x, T)
            n_disp += 1
            src = T
            if policy == "cascade":
                carry = v
                for T2 in range(T + 1, K):
                    if remaining[T2] >= size:
                        put(carry, T2)
                        moves[(src, T2)] += 1
                        return T
                    v2 = worst(T2)
                    if v2 is not None and rk(v2) > rk(carry):
                        take(v2)
                        put(carry, T2)
                        moves[(src, T2)] += 1
                        carry, src = v2, T2
                raise RuntimeError("no tier with room below a displacement")
            dest = next(k for k in range(T + 1, K) if remaining[k] >= size)
            put(v, dest)
            moves[(T, dest)] += 1
            return T
        put(x, F)
        return F

    access_cost = 0.0
    for e in events:
        b = e if e >= 0 else ~e                    # ~b: the block's creation (compulsory miss)
        if where[b] < 0:
            assert is_new[b], "a pre-split block was not placed at the split"
            k = admit(b)
            fixed_cost += fixed[k]
            admitted.append(b)
        if e >= 0:
            access_cost += per[where[b]]
    return {"access_cost": access_cost, "fixed_cost": fixed_cost, "moves": moves, "where": where,
            "n_displacements": n_disp, "admitted": admitted,
            "final_tier_counts": Counter(tiers[where[b]].value for b in range(n) if where[b] >= 0)}


def total_migration(moves: Counter, cm: dict, w, model: str) -> float:
    tiers = cm["tiers"]
    return sum(c * migration_cost(tiers[s], tiers[d], cm["read_ms"], w, model)
               for (s, d), c in moves.items())


# --- trace inputs ------------------------------------------------------------------------------

def post_split_events(path: str, st: int, max_requests: int, index: dict) -> list[int]:
    """Post-split accesses in time order, blocks in request order (as arrival_features sees them).
    Uses the same request parser as run_arrival_admission.py."""
    from train_session_features import parse_requests
    # A block first seen at or after the split is CREATED by its first event: a compulsory miss,
    # not billed (Workload.future_counts, notebook AS) -- but it is still the moment the block is
    # admitted. Such an event is encoded as ~index (negative): it admits, it costs nothing.
    seen, out = set(), []
    for t, _sid, blocks in parse_requests(path, max_requests):
        for blk in blocks:
            if t >= st:
                out.append(index[blk] if blk in seen else ~index[blk])
            seen.add(blk)
    return out


def arm_new_beliefs(path: str, wl, st: int, n_req: int) -> tuple[dict, list[str], dict]:
    """New-block beliefs for every arm, replicating run_arrival_admission.main() lines 127-235
    (feature build, split, windowed fit, quantile-matched order arms, constant magnitude arms)
    with its own helpers. Replicated rather than called because main() builds them inline; the
    replication is checked against experiments/tier_only.json by the caller."""
    from run_arrival_admission import arrival_features
    from stoa.causal import analogue_magnitudes, split_by_arrival, windowed_training_labels
    from stoa.sequential import quantile_match
    from train_session_features import fit_logistic, predict

    X, _truth, ids, arrival = arrival_features(path, n_req)   # _truth: eval labels, never used
    tr_idx, ev_idx = split_by_arrival(arrival, st)
    steps_of = {b: wl.traces[b].access_steps for b in ids}
    window = st // 2
    fit_idx, fit_lab = windowed_training_labels(arrival, ids, steps_of, st, window)
    assert all(arrival[i] + window < st for i in fit_idx)
    model = fit_logistic([X[i] for i in fit_idx], [1.0 if v > 0 else 0.0 for v in fit_lab])
    scores = predict(model, [X[i] for i in ev_idx])
    ev_ids = [ids[i] for i in ev_idx]
    pre = wl.prefix_stats(st)
    ref = analogue_magnitudes((v["count"] for v in pre.values()), len(ev_ids))

    perm = list(range(len(ev_idx)))
    random.Random(17).shuffle(perm)
    assert all(arrival[ev_idx[k]] <= arrival[ev_idx[k + 1]] for k in range(len(ev_idx) - 1))
    _ord = sorted(range(len(ev_idx)), key=lambda k: (scores[k], -k))
    learned_rank = [0.0] * len(ev_idx)
    for rnk, k in enumerate(_ord):
        learned_rank[k] = float(rnk)
    order_arms = {"order-arrival": [-float(k) for k in range(len(ev_idx))],
                  "order-learned": learned_rank,
                  "order-shuffled": [float(v) for v in perm]}
    mag_arms = {"constant-mean": statistics.mean(ref), "constant-median": statistics.median(ref)}
    out = {arm: dict(zip(ev_ids, quantile_match(list(vals), ref)))
           for arm, vals in order_arms.items()}
    out.update({arm: {i: m for i in ev_ids} for arm, m in mag_arms.items()})
    return out, ev_ids, mag_arms


# --- main --------------------------------------------------------------------------------------

def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description="Online admission control, tier-only (H6)")
    ap.add_argument("--requests", type=int, default=0, help="0 = full trace")
    ap.add_argument("--out", type=str, default=str(ROOT / "experiments" / "online_admission.json"))
    args = ap.parse_args()
    n_req = args.requests or 10 ** 9

    # Tier-only, applied exactly as run_tier_only.py applies it: before stoa.gnn is imported.
    TIER_ONLY = tier_only_actions()
    assert "stoa.gnn" not in sys.modules, "stoa.gnn imported before the tier-only patch"
    Action.space = staticmethod(lambda: list(TIER_ONLY))
    import stoa.gnn
    assert len(stoa.gnn.LEGAL_ACTIONS) == len(TIER_ONLY) == 4
    from run_arrival_admission import TIE_EPS, TRACES
    from stoa.mooncake import load_mooncake
    from stoa.sequential import attainable_ceiling, place_with_beliefs, prefix_greedy_cost

    import os
    os.chdir(ROOT)                                # TRACES paths are repo-relative
    tier_only_json = json.loads((ROOT / "experiments" / "tier_only.json").read_text())
    static_rows = {(r["trace"], r["arm"], r.get("tie")): r
                   for r in tier_only_json["arrival_admission"]["results"]}
    sim, w = TieringSimulator(), DEFAULT_WEIGHTS
    validation, rows, diagnostics = {}, [], {}

    for name, path in TRACES.items():
        wl = load_mooncake(path, max_requests=args.requests or None)
        st = max(1, wl.time_split(0.5))
        pre = wl.prefix_stats(st)
        fut = wl.future_counts(st)
        ids = [i.item_id for i in wl.items]
        index = {b: k for k, b in enumerate(ids)}
        sizes = [float(i.size_bytes) for i in wl.items]
        assert len(set(sizes)) == 1, "non-uniform block sizes"
        rep = wl.items[0].representation
        assert all(i.representation == rep for i in wl.items)
        cm = cost_model(sim, w, TIER_ONLY, rep)
        caps = sim.tier_capacities(wl.total_bytes)
        mig_rw_idx = (lambda i, j, _t=cm["tiers"], _r=cm["read_ms"]:
                      migration_cost(_t[i], _t[j], _r, w, "read+write"))
        events = post_split_events(path, st, n_req, index)
        counts = [float(pre[b]["count"]) for b in ids]
        is_new = [c == 0 for c in counts]
        assert Counter(e for e in events if e >= 0) == Counter({index[b]: c for b, c in fut.items() if c > 0}), \
            "request-order events disagree with future_counts"

        # ---- 1. validate the per-access bill against place_with_beliefs, static ----------------
        heur = {b: counts[k] for k, b in enumerate(ids)}
        orac = {b: float(fut.get(b, 0)) for b in ids}
        seen = [b for k, b in enumerate(ids) if not is_new[k]]
        c_pwb_heur = place_with_beliefs(wl, heur)
        c_pgc_heur = prefix_greedy_cost(wl)
        c_pwb_orac = place_with_beliefs(wl, orac)
        c_pwb_ceil = attainable_ceiling(wl, seen)

        def static_bill(belief_map, new_ahead=False):
            bl = [belief_map.get(b, 0.0) for b in ids]
            return bill_static(events, static_tiers(rank_order(bl, is_new, new_ahead), sizes,
                                                    caps, cm["tiers"]), cm)

        c_heur = static_bill(heur)
        c_orac = static_bill(orac)
        c_ceil = static_bill({b: (orac[b] if not is_new[k] else 0.0) for k, b in enumerate(ids)})
        # The online controller with every new block at belief 0 can never displace; it must
        # reproduce the frequency heuristic exactly.
        c_online_heur = {tie: run_online(counts, is_new, sizes[0], caps, cm, events,
                                         tie == "new-ahead", "cascade")["access_cost"]
                         for tie in TIES}
        span = c_heur - c_orac
        val = {"static_bill_vs_place_with_beliefs_rel": {
                   "heuristic": _rel(c_heur, c_pwb_heur),
                   "heuristic_vs_prefix_greedy_cost": _rel(c_heur, c_pgc_heur),
                   "oracle": _rel(c_orac, c_pwb_orac),
                   "split_ceiling": _rel(c_ceil, c_pwb_ceil)},
               "online_controller_with_new_beliefs_0_vs_heuristic_rel":
                   {t: _rel(v, c_heur) for t, v in c_online_heur.items()},
               "costs": {"heuristic": c_heur, "oracle": c_orac, "split_ceiling": c_ceil,
                         "place_with_beliefs_heuristic": c_pwb_heur,
                         "prefix_greedy_cost": c_pgc_heur, "place_with_beliefs_oracle": c_pwb_orac,
                         "attainable_ceiling": c_pwb_ceil}}
        worst_rel = max(list(val["static_bill_vs_place_with_beliefs_rel"].values())
                        + list(val["online_controller_with_new_beliefs_0_vs_heuristic_rel"].values()))
        assert worst_rel < 1e-6, f"{name}: per-access billing does not reproduce: {val}"
        ceil_pct = 100.0 * (c_heur - c_ceil) / span
        rc = tier_only_json["reachable_ceiling"]["traces"][name]["rows"][0]
        if not args.requests:
            assert abs(ceil_pct - rc["reachable_pct"]) <= 0.051, (ceil_pct, rc["reachable_pct"])

        # ---- 2. the arms' beliefs, and their static value (cross-check vs tier_only.json) -------
        arms, ev_ids, mag = arm_new_beliefs(path, wl, st, n_req)
        assert set(ev_ids) == {b for k, b in enumerate(ids) if is_new[k]}, "new-block sets differ"
        assert ev_ids == [b for k, b in enumerate(ids) if is_new[k]], "admission order != item order"

        def static_pwb(belief_for_new, tie):        # run_arrival_admission's `bill`, verbatim logic
            if tie == "new-ahead":
                belief_for_new = {k: v + TIE_EPS for k, v in belief_for_new.items()}
            b = dict(heur)
            b.update(belief_for_new)
            return place_with_beliefs(wl, b)

        print(f"\n=== {name}: {len(ids)} blocks, {len(events)} post-split accesses, "
              f"{sum(is_new)} new; split ceiling {ceil_pct:.2f}%; billing residual {worst_rel:.1e}")
        print(f"{'arm':<16} {'tie':<10} {'policy':<9} {'static':>7} {'online0':>8} {'onlineRW':>9} "
              f"{'ratio0':>7} {'ratioRW':>7} {'disp':>7} {'mig RW':>9}")
        arm_val = {}
        trace_ceiling_online = {}
        for tie in TIES:
            ahead = tie == "new-ahead"
            for arm in ARMS:
                nb = arms[arm]
                c_static_pwb = static_pwb(nb, tie)
                c_static_mine = static_bill({**heur, **nb}, new_ahead=ahead)
                s_pct = 100.0 * (c_heur - c_static_pwb) / span
                ref_row = static_rows[(name, arm, tie)]
                arm_val[f"{arm}|{tie}"] = {
                    "static_mine_vs_place_with_beliefs_rel": _rel(c_static_mine, c_static_pwb),
                    "static_pct_recomputed": s_pct,
                    "static_pct_tier_only_json": ref_row["captured_pct"]}
                assert _rel(c_static_mine, c_static_pwb) < 1e-6, (arm, tie)
                if not args.requests:   # tier_only.json is a full-trace artifact
                    assert abs(s_pct - ref_row["captured_pct"]) <= 0.006, (arm, tie, s_pct, ref_row)
                bl = [nb.get(b, counts[k]) if is_new[k] else counts[k] for k, b in enumerate(ids)]
                tier_static = static_tiers(rank_order(bl, is_new, ahead), sizes, caps, cm["tiers"])
                st_pct = ref_row["captured_pct"] if not args.requests else round(s_pct, 2)
                for policy in POLICIES + ("costaware",):
                    models = ("read+write",) if policy == "costaware" else ("zero", "read+write")
                    r = run_online(bl, is_new, sizes[0], caps, cm, events, ahead, policy,
                                   mig_fn=mig_rw_idx)
                    assert [ids[b] for b in r["admitted"]] == ev_ids
                    if policy == "cascade":
                        # Cascade keeps every tier ranked ahead of the next, so once every block
                        # has arrived it must hold exactly the static assignment.
                        assert [cm["tiers"][k] for k in r["where"]] == tier_static, \
                            "cascade's final placement differs from the static assignment"
                    mig = {m: total_migration(r["moves"], cm, w, m) for m in models}
                    pct = {m: 100.0 * (c_heur - (r["access_cost"] + r["fixed_cost"] + mig[m])) / span
                           for m in models}
                    ratio = {m: (pct[m] / st_pct if st_pct > 0 else None) for m in models}
                    # Break-even migration price, as a multiple of the read+write assumption, at
                    # which this run falls to 70% of the static share / to 3x the split ceiling.
                    # Defined only where decisions do not depend on the price (not costaware).
                    mrw = 100.0 * total_migration(r["moves"], cm, w, "read+write") / span
                    p0 = 100.0 * (c_heur - r["access_cost"] - r["fixed_cost"]) / span
                    be = {}
                    if policy in ("onestep", "cascade"):
                        be = {"alpha_at_70pct_of_static": (round((p0 - 0.7 * st_pct) / mrw, 4)
                                                           if mrw > 0 and st_pct > 0 else None),
                              "alpha_at_3x_split_ceiling": (round((p0 - 3 * rc["reachable_pct"])
                                                                  / mrw, 4) if mrw > 0 else None)}
                    for m in models:
                        rows.append({
                            "trace": name, "arm": arm, "tie": tie, "policy": policy,
                            "migration_cost_model": m,
                            "new_block_belief": round(mag[arm], 4) if arm in mag else None,
                            "online_captured_pct": round(pct[m], 2),
                            "static_captured_pct": st_pct,
                            "online_over_static": round(ratio[m], 4) if ratio[m] is not None else None,
                            "split_ceiling_pct": rc["reachable_pct"],
                            "online_over_split_ceiling": round(pct[m] / rc["reachable_pct"], 3),
                            "n_displacements": r["n_displacements"],
                            "n_migrations": sum(r["moves"].values()),
                            "migrations_by_tier": {f"{cm['tiers'][a].value}->{cm['tiers'][d].value}": c
                                                   for (a, d), c in sorted(r["moves"].items())},
                            "migration_cost": round(mig[m], 3),
                            "access_cost": round(r["access_cost"], 3),
                            "full_gap": round(span, 3), **be})
                    f = lambda v: f"{v:7.2f}" if v is not None else "    n/a"   # noqa: E731
                    print(f"{arm:<16} {tie:<10} {policy:<9} {st_pct:7.2f} "
                          f"{pct['zero'] if 'zero' in pct else float('nan'):8.2f} "
                          f"{pct['read+write']:9.2f} {f(ratio.get('zero'))} {f(ratio['read+write'])} "
                          f"{r['n_displacements']:7d} {mig['read+write']:9.1f}  "
                          f"a70={be.get('alpha_at_70pct_of_static')} a3x={be.get('alpha_at_3x_split_ceiling')}")
            # Diagnostic: the clairvoyant arrival ceiling (true future counts for new blocks), run
            # through the same online controller. Reads the future by definition; never an arm.
            clair = [orac[b] if is_new[k] else counts[k] for k, b in enumerate(ids)]
            for policy in POLICIES + ("costaware",):
                r = run_online(clair, is_new, sizes[0], caps, cm, events, ahead, policy,
                               mig_fn=mig_rw_idx)
                mig_rw = total_migration(r["moves"], cm, w, "read+write")
                trace_ceiling_online[f"{tie}|{policy}"] = {
                    "online_pct_zero_mig": (None if policy == "costaware" else round(
                        100.0 * (c_heur - r["access_cost"] - r["fixed_cost"]) / span, 2)),
                    "online_pct_rw_mig": round(100.0 * (c_heur - r["access_cost"] - r["fixed_cost"]
                                                        - mig_rw) / span, 2),
                    "n_displacements": r["n_displacements"]}
        static_arrival_ceiling = static_rows[(name, "arrival-ceiling", None)]["captured_pct"]
        validation[name] = {**val, "arms_static": arm_val}
        diagnostics[name] = {
            "n_blocks": len(ids), "n_new_blocks": sum(is_new),
            "n_post_split_accesses": len(events), "split_step": st,
            "tier_capacity_blocks": {t.value: (caps[t] / sizes[0]) for t in cm["tiers"]},
            "split_ceiling_pct_recomputed": round(ceil_pct, 3),
            "arrival_ceiling_static_pct": static_arrival_ceiling,
            "arrival_ceiling_online": trace_ceiling_online}

    tiers = cm["tiers"]
    out = {
        "config": {
            "requests": args.requests or "full", "split_frac": 0.5,
            "actions": [a.tier.value for a in TIER_ONLY], "representation": "plaintext",
            "timing": "online",
            "per_access_cost": {t.value: cm["per"][t] for t in tiers},
            "fixed_cost_per_block": {t.value: cm["fixed"][t] for t in tiers},
            "migration_cost_read+write": {f"{s.value}->{d.value}": migration_cost(
                s, d, cm["read_ms"], w, "read+write") for i, s in enumerate(tiers)
                for d in tiers[i + 1:]},
            "break_even_alpha": "alpha_at_70pct_of_static / alpha_at_3x_split_ceiling: the multiple "
                                "of the read+write migration price at which a run's captured share "
                                "falls to 70% of the static arm's / to 3x the split ceiling (decisions "
                                "of onestep/cascade/literal do not depend on the price, so this is "
                                "exact for them).",
            "denominator": "heuristic static cost - oracle static cost, both per-access billed "
                           "(validated equal to place_with_beliefs, tier-only)",
            "static_values_from": "experiments/tier_only.json (re-derived here and asserted "
                                  "within 0.006 points)",
            "split_ceiling_from": "experiments/tier_only.json reachable_ceiling (recomputed and "
                                  "asserted within 0.051 points of the rounded value)",
            "policies": {
                "costaware": "as onestep, but displace only if the believed saving "
                             "b_new*(c[F]-c[T]) - b_victim*(c[F]-c[T]) exceeds the read+write "
                             "migration cost; run with read+write pricing only.",
                "onestep": "a new block takes the fastest tier with room unless a FASTER (full) "
                           "tier's lowest-ranked resident ranks behind it; then it takes that "
                           "slot and the resident is demoted to the fastest slower tier with room "
                           "(one migration). Primary.",
                "cascade": "as onestep, but the demoted resident is re-admitted below, displacing "
                           "a resident it outranks there in turn; each move is one migration.",
                "literal": "displacement considered only when the only tier with room is the "
                           "slowest (DISK), as the task statement reads literally."},
            "ties": {"new-behind": "a new block does NOT displace a pre-split resident of equal "
                                   "belief (new loses ties)",
                     "new-ahead": "a new block displaces a pre-split resident of equal belief "
                                  "(new wins ties)",
                     "between_new_blocks": "the earlier-admitted block wins at equal belief in "
                                           "both conventions (the static pass's stable order)"}},
        "assumptions": [
            "Per-access cost = slope of weighted_cost(cost_of_placement(a, n)) in n (asserted "
            "affine); fixed cost = intercept, which is 0 for every tier under PLAINTEXT and is "
            "charged once at a block's first placement, as place_with_beliefs charges it.",
            "The token term (lambda_B x 800 tokens = 0.4 per access) is identical on every tier "
            "and cancels in every captured share; it is kept so absolute costs match "
            "place_with_beliefs.",
            "The simulator defines NO tier-migration cost (its one-off cost is a representation "
            "change, 0 for PLAINTEXT). Migration is therefore assumed to cost one read from the "
            "source tier plus one write to the destination, with the write priced at the "
            "destination's read latency, weighted by lambda_L, no token term "
            "(read+write). A zero-migration-cost variant is reported beside it.",
            "Initial placement (pre-split blocks at the split, a new block at admission) is free, "
            "as in place_with_beliefs.",
            "Beliefs are fixed once assigned: pre-split blocks keep their prefix count, new "
            "blocks keep the arm's belief; nothing is promoted on later evidence.",
            "Block sizes are uniform (asserted), so one demotion frees exactly one slot.",
            "Displacement is decided by belief rank only; it does not weigh the migration cost "
            "against the expected saving.",
            "Within a request, blocks are admitted in request (position) order; this is the "
            "order arrival_features assigns arrival indices in, and asserted equal to item order.",
            "New-block beliefs replicate run_arrival_admission.main() (its helpers are imported; "
            "the inline arm construction is copied), and each arm's static value is re-derived "
            "and asserted equal to tier_only.json."],
        "validation": validation,
        "diagnostics": diagnostics,
        "rows": rows,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
