#!/usr/bin/env python3
"""Check every headline number in the paper against the artifact it came from.

The tenth defect in the paper's own catalog was believing an edit script's success message
instead of checking the file. This script is that lesson applied to the paper: it reads the
JSON artifacts, recomputes each claimed quantity, and greps the LaTeX source for the string
that should carry it. A number that drifts when an experiment is re-run -- which is exactly
what happens when a script's defaults change or an artifact is regenerated at a new scale --
fails here rather than reaching a reviewer.

It checks the SOURCE, not the log or the PDF text, because the PDF can be stale and the log
reports success for edits that matched nothing.

Exit status is 0 when every check passes, 1 otherwise.
Usage:  python3 scripts/check_paper_numbers.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
import re

_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from stoa import verify  # noqa: E402

_RAW = " ".join(p.read_text() for p in
                [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))])
# Collapse runs of whitespace. LaTeX table columns are padded for readability in the source,
# so an otherwise-correct needle fails on cosmetic alignment -- which teaches you to ignore
# the checker. Whitespace is the one difference that never changes a number.
TEX = re.sub(r"\s+", " ", _RAW)

failures: list[str] = []
checked = 0


_LOADED: list[str] = []


# Results the ICDE version deliberately omits: they depend on a paid hosted model, and ICDE's
# EAB category requires every result to be reproducible from the artifact, "no exceptions". The
# artifacts stay in experiments/ and their claims stay in this file for the record (they back
# the PVLDB version, git tag pvldb-final-2026-09); they are skipped by name, loudly, rather than
# deleted. See docs/claims_dependency.md §AO.
NOT_IN_THIS_PAPER = {"eval_locomo_powered.json", "representation_axis.json",
                     "locomo_power.json", "eval_locomo_leakfree.json"}


def load(name):
    """Read an artifact, recording the name so the staleness map can be audited against it."""
    if name in NOT_IN_THIS_PAPER:
        print(f"\n[skip] {name}: not in this paper (LLM-dependent; omitted for ICDE EAB)")
        return None
    _LOADED.append(name)
    p = EXP / name
    return json.loads(p.read_text()) if p.exists() else None


_ASSERTED: list[str] = []


def claim(label: str, needle: str, derived, source: str) -> None:
    """Assert `needle` appears in the LaTeX source, and report what the artifact says.

    `derived` is the value recomputed from the artifact, printed alongside so a mismatch is
    diagnosable rather than just red.
    """
    global checked
    checked += 1
    _ASSERTED.append(str(needle))
    ok = re.sub(r"\s+", " ", needle) in TEX
    print(f"  [{'ok ' if ok else 'MISS'}] {label:<52} {derived}   ({source})")
    if not ok:
        failures.append(f"{label}: expected {needle!r} in the paper source, from {source}")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


print("Checking paper claims against experiments/*.json\n")

# --- reactive tiering + ARC + LRB -------------------------------------------------
d = load("reactive_real_full_lrb.json")   # no fallback: reactive_real_full.json has no target
if d:
    print("reactive (experiments/reactive_real_full_lrb.json)")
    for trace in ("toolagent", "conversation"):
        rows = d["traces"][trace]["rows"]
        vals = [r["best_reactive_over_belady"] for r in rows]
        # Derived, not hardcoded: an earlier version pinned "75--99" as a literal and went
        # stale the moment two more policies entered the envelope.
        claim(f"{trace} envelope (all policies)", f"{pct(min(vals))}--{pct(max(vals))}",
              f"{pct(min(vals))}-{pct(max(vals))}%", "best_reactive_over_belady")
        # The abstract calls one of these bands "forecast nothing". Only the pure-rule band
        # qualifies: at a 2% tier the full envelope is attained by LeCaR, a learner.
        pure = [max(r["lru"], r["h2o"], r["static"]) / r["belady"] for r in rows]
        claim(f"{trace} envelope (pure rules only)",
              f"{pct(min(pure))}--{pct(max(pure))}",
              f"{pct(min(pure))}-{pct(max(pure))}%", "lru/h2o/static only")
        # Reconstruct each LaTeX table row exactly as the paper should print it. Checking the
        # whole row rather than each number separately is what makes this a real check: a
        # lone "29.4" could match anything on the page.
        for r in rows:
            b = r["belady"]
            # LRB is deliberately absent: its repaired value is a band, reported in its own
            # paragraph. Asserting the broken point value here would re-import the retracted
            # number through the checker.
            cells = [max(r["lru"], r["h2o"], r["static"]) / b] + [
                r[k] / b for k in ("arc", "lecar", "cacheus")]
            best = max(cells)
            row = " & ".join((f"\\textbf{{{pct(c, 1)}}}" if c == best else pct(c, 1))
                             for c in cells)
            claim(f"{trace} {r['cache_frac']:.0%} table row", row, row, "reactive row")

    # The adaptive/learned family vs the best pure rule -- the paper's "at most 4.6 points".
    gains = []
    for t in d["traces"].values():
        for r in t["rows"]:
            b = r["belady"]
            pure = max(r["lru"], r["h2o"], r["static"]) / b
            soph = max(r[k] / b for k in ("arc", "lecar", "cacheus"))
            gains.append(soph - pure)
    claim("max gain of any adaptive/learned policy", f"{100 * max(gains):.1f} points",
          f"{100 * max(gains):+.1f} pts (min {100 * min(gains):+.1f})", "arc/lecar/cacheus/lrb")
    ghosts = [r["arc_ghost_hit_rate_of_misses"]
              for t in d["traces"].values() for r in t["rows"]]
    claim("ARC ghost-hit rate range", f"{pct(min(ghosts), 1)}--{pct(max(ghosts), 1)}",
          f"{pct(min(ghosts), 1)}-{pct(max(ghosts), 1)}%", "arc_ghost_hit_rate_of_misses")

# --- LoCoMo power analysis --------------------------------------------------------
# §5.7's sample sizes reproduce exactly from the powered run's stored discordance counts --
# they were correct all along and simply had no assertion, which is how a number survives
# twelve passes unwatched. Derived here rather than regenerated: the artifact holds the raw
# counts, and the paid LLM run that produced it need not be repeated to check arithmetic.
d = load("eval_locomo_powered.json")
if d:
    from stoa.stats import n_for_power_mcnemar
    print("\nLoCoMo required sample size (experiments/eval_locomo_powered.json)")
    thr, n_arm = d["bonferroni_threshold"], 300
    sizes, sizes_uncorr = [], []
    for c in d["contrasts"]:
        if c.get("high") != "stoa" or c.get("low") != "random":
            continue
        hi, lo = c["discordant_high"], c["discordant_low"]
        p_disc, p_fav = (hi + lo) / n_arm, hi / (hi + lo)
        sizes.append(n_for_power_mcnemar(p_disc, p_fav, alpha=thr, seed=0))
        sizes_uncorr.append(n_for_power_mcnemar(p_disc, p_fav, alpha=0.05, seed=0))
    if len(sizes) == 3:
        claim("n per arm at the corrected threshold",
              f"{sizes[0]}, {sizes[1]} and {sizes[2]} questions per arm",
              f"alpha={thr}", "n_for_power_mcnemar on discordant counts")
        claim("n per arm at an uncorrected 0.05",
              f"{sizes_uncorr[0]}, {sizes_uncorr[1]} and {sizes_uncorr[2]}",
              "alpha=0.05", "the sizing mistake an earlier revision made")

d = load("locomo_power.json")
if d:
    print("\nLoCoMo power (experiments/locomo_power.json)")
    # The pilot is now background: the paper cites it for the sample size it demanded, which
    # is what motivated the re-run. Numbers the paper no longer states are not checked --
    # a checker that asserts absent claims trains you to ignore it.
    surv = sum(r["n_surviving_correction"] for r in d["runs"].values())
    assert surv == 0, "pilot survivors changed; the paper says none survived"
    ns = [r["n_per_arm_for_80pct_power"] for r in d["power"]
          if r["high"] == "stoa" and r["low"] == "random" and r["n_per_arm_for_80pct_power"]]
    claim("pilot: n demanded for placement vs random", f"{min(ns)}--{max(ns)}",
          f"{min(ns)}-{max(ns)}", "n_per_arm_for_80pct_power")

# --- hardware-derived cost-model sweep --------------------------------------------
d = load("calibration_sensitivity.json")
if d:
    print("\ncalibration sweep (experiments/calibration_sensitivity.json)")
    s2 = d["summary"]
    claim("derived headroom range", f"{s2['headroom_min']:.0f}--{s2['headroom_max']:.0f}\\%",
          f"{s2['headroom_min']:.1f}-{s2['headroom_max']:.1f}%", "summary")
    claim("hand-written table headroom", f"{s2['handwritten_headroom']:.0f}\\%",
          f"{s2['handwritten_headroom']:.1f}%", "handwritten_headroom")
    n_nm, n_tot = len(s2["non_monotonic"]), s2["n_models"] * s2["n_devices"]
    claim("non-monotonic device classes", f"{n_nm} of the {n_tot}",
          f"{n_nm}/{n_tot}", "non_monotonic")

# --- LoCoMo at adequate scale (paired tests) ---------------------------------------
d = load("eval_locomo_powered.json")
if d:
    print("\nLoCoMo powered (experiments/eval_locomo_powered.json)")
    c = d["config"]
    claim("scored questions per arm", str(c["n_scored_per_arm"]),
          str(c["n_scored_per_arm"]), "n_scored_per_arm")
    for i, f in enumerate(d["budget_fractions"]):
        row = " & ".join(
            (f"\\textbf{{{d['accuracy'][m][i] * 100:.1f}}}"
             if d["accuracy"][m][i] == max(d["accuracy"][k][i] for k in d["accuracy"])
             else f"{d['accuracy'][m][i] * 100:.1f}")
            for m in ("stoa", "random", "centrality", "retrieval"))
        claim(f"accuracy row @{f:.0%}", row, row, "accuracy")
    words = {4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
    n_s, n_c = d["n_surviving_correction"], len(d["contrasts"])
    claim("comparisons surviving correction",
          f"{words.get(n_s, str(n_s))} of the {words.get(n_c, str(n_c)).lower()} comparisons",
          f"{n_s}/{n_c}", "n_surviving_correction")
    # retrieval must lead in every dialogue at every budget for the "10 of 10" claim
    r = [x for x in d["contrasts"] if x["high"] == "retrieval"]
    lo = min(x["dialogues_favouring_high"] for x in r)
    claim("dialogues favouring retrieval", f"{lo} of 10", f"{lo}/10", "dialogues_favouring_high")
    unresolved = [x for x in d["contrasts"]
                  if x["low"] == "random" and x["mcnemar_p"] >= d["bonferroni_threshold"]]
    ps = sorted(x["mcnemar_p"] for x in unresolved)
    if len(ps) == 3:
        claim("unresolved p-values", f"$p = {ps[0]:.3f}$, ${ps[1]:.3f}$ and ${ps[2]:.3f}$",
              f"{ps[0]:.3f}/{ps[1]:.3f}/{ps[2]:.3f}", "mcnemar_p")

# --- representation axis ------------------------------------------------------------
d = load("representation_axis.json")
if d:
    print("\nrepresentation axis (experiments/representation_axis.json)")
    for r in d["rows"]:
        cells = [r["acc_plaintext"], r["acc_summary"], r["acc_extract"]]
        best = max(cells)
        row = " & ".join((f"\\textbf{{{c * 100:.1f}}}" if c == best else f"{c * 100:.1f}")
                         for c in cells)
        claim(f"accuracy row @{r['budget_frac']:.0%}", row, row, "rows")
    cr = d["compression_ratio"]
    claim("compression ratios", f"summary ${cr['summary']:.2f}\\times$, extract "
                                f"${cr['extract']:.2f}\\times$",
          f"summary {cr['summary']:.2f}, extract {cr['extract']:.2f}", "compression_ratio")
    thr = 0.05 / len(d["contrasts"])
    surv = sum(1 for c in d["contrasts"] if c["mcnemar_p"] < thr)
    assert surv == 0, "a representation comparison now survives; update the paper's claim"
    best_p = min(c["mcnemar_p"] for c in d["contrasts"])
    claim("best raw p", f"raw $p = {best_p:.3f}$", f"{best_p:.3f}", "mcnemar_p")
    claim("correction threshold", f"{thr:.5f}", f"{thr:.5f}", "bonferroni")
    disc = [f"{c['discordant_mode']}--{c['discordant_plaintext']}"
            for c in d["contrasts"] if c["mode"] == "summary"]
    claim("summary discordant pairs", ", ".join(disc), ", ".join(disc), "discordant")

# --- sampling study: no parametric interval may be claimed at n=4 --------------------
d = load("sampling_study.json")
if d:
    print("\nsampling study (experiments/sampling_study.json)")
    for r in d["rows"]:
        claim(f"observed range at {r['convs']} conversations",
              f"${r['captured_observed_min']:.0f}$ and $+{r['captured_observed_max']:.0f}$",
              f"[{r['captured_observed_min']:+.0f}, {r['captured_observed_max']:+.0f}]",
              "captured_observed_min/max")
        n = r["draws"]
        assert n < 30, "with this many draws a parametric interval would be defensible again"
    # Guard: the discarded z-based figures must not reappear anywhere in the paper.
    for stale in ("$\\pm126$ points", "$\\pm660$"):
        if stale in TEX and "earlier revision" not in TEX:
            failures.append(f"stale z-based interval {stale!r} still asserted in the paper")

# A checker that passes when it checked nothing is worse than no checker: it stamps a
# paper as verified on a machine where the artifacts were never generated. Require both a
# minimum number of checks and that every artifact the paper depends on is present.

# --- arrival-time admission: conventions vs an estimated prior (V-C; notebook AP-AS) ----------
d = load("arrival_admission.json")
_to = load("tier_only.json")
_hz, _hzt = load("hazard_prior.json"), load("hazard_prior_tier.json")
_rc = load("reachable_ceiling.json")
_dg = load("arrival_diagnostics.json")
_eh = load("equal_horizon.json")
if d and _to and _rc and _dg and _hz and _hzt and _eh:
    print("\narrival admission (arrival_admission, tier_only, hazard_prior*)")
    RES = {"all": d["results"], "tier": _to["arrival_admission"]["results"]}
    HZ = {"all": _hz["rows"], "tier": _hzt["rows"]}

    def _ceil(acts, t):
        src = _rc if acts == "all" else _to["reachable_ceiling"]
        return [r for r in src["traces"][t]["rows"] if r["split_frac"] == 0.5][0]["reachable_pct"]

    def cap(acts, tie, t, arm):
        return [r["captured_pct"] for r in RES[acts] if r["trace"] == t and r.get("tie") == tie
                and r["arm"] == arm][0]

    def hz(acts, tie, t, arm):
        return [r["captured_pct"] for r in HZ[acts] if r["trace"] == t and r["tie"] == tie
                and r["arm"] == arm][0]

    def _f1(v):
        return f"{v:.1f}".replace("-0.0", "0.0")

    ORDER = ("order-arrival", "order-learned", "order-shuffled")
    PROD = ("conversation", "toolagent")
    short = {"conversation": "conv", "toolagent": "tool"}
    CFG = [(a, tie) for a in ("all", "tier") for tie in ("new-behind", "new-ahead")]
    for a, tie in CFG:
        for t in PROD:
            o = [cap(a, tie, t, x) for x in ORDER]
            row = (f"& {short[t]} & {_ceil(a, t):.1f} & {_f1(min(o))}--{_f1(max(o))} & "
                   f"{_f1(cap(a, tie, t, 'constant-mean'))} & {_f1(cap(a, tie, t, 'constant-median'))} & "
                   f"{_f1(hz(a, tie, t, 'hazard-new'))} & {_f1(hz(a, tie, t, 'hazard-all'))} \\\\")
            claim(f"{a}/{tie} {t} row", row, "", "arrival row")
    # The estimate is tie-invariant by construction; assert it rather than trust it.
    for a in ("all", "tier"):
        for r in HZ[a]:
            other = hz(a, "new-ahead" if r["tie"] == "new-behind" else "new-behind", r["trace"], r["arm"])
            assert abs(r["captured_pct"] - other) < 1e-9, "the hazard arm depends on the tie convention"
    est = _hz["estimates"]
    claim("new-block expected reuse", f"({est['conversation']['h_new']:.2f} and {est['toolagent']['h_new']:.2f})",
          "", "h_new")
    claim("seen-once expected reuse", f"({est['conversation']['m_by_count_bin'][0]:.2f} and "
          f"{est['toolagent']['m_by_count_bin'][0]:.2f})", "", "m(count 1)")
    assert all(2 < est[t]["c_star"] < 3 for t in PROD), "the paper places new blocks between count 2 and 3"
    conv_arms = [cap(a, tie, t, x) for a, tie in CFG for t in PROD
                 for x in ORDER + ("constant-mean", "constant-median")]
    claim("convention span", f"span {_f1(min(conv_arms))}--{_f1(max(conv_arms))}\\% of the advantage", "", "all arms")
    tb = [cap("tier", "new-behind", t, x) for t in PROD for x in ORDER]
    tbc = [_ceil("tier", t) for t in PROD]
    assert all(cap("tier", "new-behind", t, x) < 1.3 * _ceil("tier", t) for t in PROD for x in ORDER), \
        "the paper says tier-only/behind order arms land at the split ceiling"
    claim("tier/behind at the ceiling", f"{_f1(min(tb))}--{_f1(max(tb))}\\% against {min(tbc):.1f}--{max(tbc):.1f}\\%",
          "", "tier, new-behind")
    hz_vals = [(hz(a, "new-behind", t, x), _ceil(a, t)) for a in ("all", "tier") for t in PROD
               for x in ("hazard-new", "hazard-all")]
    r_new = [hz(a, "new-behind", t, "hazard-new") / _ceil(a, t) for a in ("all", "tier") for t in PROD]
    r_all = [v / c_ for v, c_ in hz_vals]
    claim("hazard-new ratio, V-C", f"captures {min(r_new):.1f}--{max(r_new):.1f} times the matching", "", "hazard-new / ceiling")
    claim("hazard ratio incl. old blocks", f"({min(r_all):.1f}--{max(r_all):.1f} when old blocks", "", "both hazard arms")
    claim("hazard-new ratio, abstract", f"captures {min(r_new):.1f}--{max(r_new):.1f} times what", "", "rounded")
    claim("hazard-new ratio, intro", f"captures {min(r_new):.1f} to {max(r_new):.1f}\ntimes", "", "rounded")
    claim("hazard-new ratio, C2", f"captures {min(r_new):.1f}--{max(r_new):.1f} times\n  the matching", "", "rounded")
    claim("hazard-new ratio, conclusion", f"{min(r_new):.1f}--{max(r_new):.1f} times the split-time ceiling in a static", "", "rounded")
    claim("hazard values vs ceilings", f"{min(v for v, _ in hz_vals):.1f}--{max(v for v, _ in hz_vals):.1f}\\%\n"
          f"against {min(c_ for _, c_ in hz_vals):.1f}--{max(c_ for _, c_ in hz_vals):.1f}\\%", "", "hazard arms")
    assert abs(hz("all", "new-behind", "toolagent", "hazard-new") - cap("all", "new-behind", "toolagent", "constant-mean")) < 0.05, \
        "the paper says the hazard arm coincides with the mean arm on toolagent"
    claim("abstract: new vs seen-once reuse", f"is {est['conversation']['h_new']:.2f} against "
          f"{est['conversation']['m_by_count_bin'][0]:.2f})", "", "h_new, m(1)")
    _ar, _tr = load("arrival_admission_reuse.json"), load("tier_only_reuse.json")
    if _ar and _tr:
        _rr, _rv = [], []
        for _res, _acts in ((_ar["results"], "all"), (_tr["arrival_admission"]["results"], "tier")):
            for r in _res:
                if r["trace"] in PROD and r["arm"] in ORDER:
                    _rr.append(r["captured_pct"] / _ceil(_acts, r["trace"]))
                    _rv.append(r["captured_pct"])
        claim("reuse-scale order arms", f"puts every order arm at {min(_rr):.1f}--{max(_rr):.1f} times the split-time ceiling\n"
              f"({min(_rv):.1f}--{max(_rv):.1f}\\%)", "", "reference multiset minus creation")
    _hzr, _hztr = load("hazard_prior_rescaled.json"), load("hazard_prior_tier_rescaled.json")
    if _hzr and _hztr:
        _dv = []
        for _x, _y in ((_hz, _hzr), (_hzt, _hztr)):
            _yy = {(r["trace"], r["arm"], r["tie"]): r["captured_pct"] for r in _y["rows"]}
            _dv += [abs(r["captured_pct"] - _yy[(r["trace"], r["arm"], r["tie"])]) for r in _x["rows"]
                    if r["trace"] in PROD and r["arm"] == "hazard-new"]
        claim("hazard rescale sensitivity", f"by at most {max(_dv):.1f} points on the production traces", "", "rescaled c*")
    _oa = load("online_admission.json")
    if _oa:
        prod_oa = [r for r in _oa["rows"] if r["trace"] in PROD and r["arm"] != "arrival-ceiling"]
        free = max(r["online_captured_pct"] for r in prod_oa if r["migration_cost_model"] == "zero")
        priced = max(r["online_captured_pct"] for r in prod_oa if r["migration_cost_model"] == "read+write")
        assert priced <= 0.0 + 1e-9, "the paper says no arm beats the heuristic once a move costs a read and a write"
        claim("online, free migration", f"replayed online (at most {free:.1f}\\%)", "", "online_admission")
        _h70 = {r["trace"]: r["alpha_at_70pct_of_static"] for r in _oa["rows"] if r["trace"] in PROD
                and r["arm"] == "hazard-new" and r["policy"] == "onestep" and r["migration_cost_model"] == "read+write"
                and r["tie"] == "new-behind"}
        claim("hazard break-even", f"at most {100*_h70['conversation']:.1f}\\% and {100*_h70['toolagent']:.1f}\\% of that price",
              "", "alpha_at_70pct_of_static")
        _hzon = [r["online_captured_pct"] for r in _oa["rows"] if r["trace"] in PROD and r["arm"] == "hazard-new"
                 and r["migration_cost_model"] == "read+write"]
        assert max(_hzon) <= 1e-9, "the paper says the hazard arm does not beat the heuristic with priced moves"
    _rcst = load("reactive_cost.json")
    if _rcst:
        _rr = {r["trace"]: r["reactive"] for r in _rcst["rows"]}
        _lz = [_rr[t]["lru"]["zero"] for t in PROD]
        claim("LRU on the cost metric, free", f"captures {_lz[0]:.1f}\\% and {_lz[1]:.1f}\\%", "", "reactive_cost")
        claim("second-hit on the cost metric", f"captures\n${_rr['conversation']['second-hit']['zero']:.1f}$\\% and "
              f"${_rr['toolagent']['second-hit']['zero']:.1f}$\\%", "", "reactive_cost")
        claim("LRU priced", f"LRU falls to ${_rr['conversation']['lru']['read+write']:.1f}$\\% and\n"
              f"${_rr['toolagent']['lru']['read+write']:.1f}$\\%", "", "reactive_cost")
        assert all(_rr[t]["none"]["zero"] == 0.0 for t in _rr), "the replay no longer reproduces the heuristic"
        assert min(_lz) > free, "the paper says LRU beats every online arrival arm with free moves"
        # "none of the policies we replay that move blocks beats the static frequency heuristic" once a
        # move costs one read plus one write: every replayed arm, policy and tie, and both reactive policies
        _priced = [r["online_captured_pct"] for r in _oa["rows"] if r["trace"] in PROD
                   and r["migration_cost_model"] == "read+write"]
        _priced += [_rr[t][pol]["read+write"] for t in PROD for pol in ("lru", "second-hit")]
        assert max(_priced) <= 1e-9, f"a replayed policy beats the heuristic with priced moves ({max(_priced)})"
    shift = [cap(a, "new-ahead", t, x) - cap(a, "new-behind", t, x) for a in ("all", "tier")
             for t in PROD for x in ORDER]
    claim("tie shift", f"by {min(shift):.0f}--{max(shift):.0f} points", "", "ahead - behind")
    claim("tie shift, intro", f"by {min(shift):.0f} to {max(shift):.0f} points", "", "ahead - behind")
    spread = max(max(cap(a, tie, t, x) for x in ORDER) - min(cap(a, tie, t, x) for x in ORDER)
                 for a, tie in CFG for t in PROD)
    claim("order spread", f"at most {spread:.1f}", "", "max-min of order arms")
    worst = max(max(cap(a, tie, t, x) for a, tie in CFG) - min(cap(a, tie, t, x) for a, tie in CFG)
                for t in PROD for x in ORDER + ("constant-mean", "constant-median"))
    claim("convention sensitivity of one arm", f"up to {worst:.0f} points", "", "max over configs")
    sy = [cap(a, tie, "synthetic", x) for a, tie in CFG for x in ORDER + ("constant-mean", "constant-median")]
    sy += [hz(a, "new-behind", "synthetic", x) for a in ("all", "tier") for x in ("hazard-new", "hazard-all")]
    assert max(cap("all", tie, "synthetic", x) for tie in ("new-behind", "new-ahead")
               for x in ORDER + ("constant-mean", "constant-median")) < _ceil("all", "synthetic")
    claim("synthetic", f"{_ceil('all', 'synthetic'):.1f}\\% ({_ceil('tier', 'synthetic'):.1f}\\% tier-only) and every\n"
          f"arrival arm, the hazard arms included, stays below it ({_f1(min(sy))}--{_f1(max(sy))}\\%)", "", "synthetic")
    rows_dg = {r["trace"]: r for r in _dg["rows"]}
    ratio = [r["first_over_last"] for r in _dg["rows"]]
    aucs = [r["auc_arrival_order"] for r in _dg["rows"]]
    _pr = [rows_dg[t] for t in PROD]
    _pratio = sorted(r["first_over_last"] for r in _pr)
    _pauc = sorted(r["auc_arrival_order"] for r in _pr)
    claim("exposure ratio, production", f"({_pratio[0]:.1f}--{_pratio[-1]:.1f} times", "", "diagnostics")
    claim("arrival-order AUC, production", f"AUC {_pauc[0]:.2f}--{_pauc[-1]:.2f} only because", "", "diagnostics")
    eh = {r["trace"]: r for r in _eh["rows"]}
    claim("equal-horizon AUC", f"its AUC is {eh['conversation']['auc_order_equal_horizon']:.2f} and "
          f"{eh['toolagent']['auc_order_equal_horizon']:.2f}", "", "equal_horizon")
    claim("old design's label leak", f"({rows_dg['conversation']['old_design_label_mass_after_cut_pct']:.1f}\\% and "
          f"{rows_dg['toolagent']['old_design_label_mass_after_cut_pct']:.1f}\\% of label", "", "diagnostics")
    claim("split ceiling, introduction", f"(\\textbf{{{_ceil('all', 'conversation'):.1f}\\%}} and "
          f"\\textbf{{{_ceil('all', 'toolagent'):.1f}\\%}} of", "", "reachable_ceiling @ 0.5")

# External replication on Alibaba's Bailian traces (V-C; notebook AT, AU).
_bR, _bH = load("bailian_reachable_birthx.json"), load("bailian_hazard_birthx.json")
if _bR and _bH:
    print("\nexternal traces (bailian_reachable_birthx.json, bailian_hazard_birthx.json)")
    _BT = ("traceA", "traceB", "coder", "thinking")
    _un = [[r["unreachable_pct"] for r in _bR["full"]["traces"][t]["rows"] if r["split_frac"] == 0.5][0] for t in _BT]
    _rt = {sp: {r["trace"]: r["captured_pct"] / r["split_ceiling_pct"] for r in _bH["rows"]
                if r["actions"] == sp and r["arm"] == "hazard-new" and r["tie"] == "new-behind"} for sp in ("full", "tier_only")}
    _tpos = [v for v in _rt["tier_only"].values() if v > 0.01]
    assert sum(1 for v in _rt["tier_only"].values() if v <= 0.01) == 1 and _rt["tier_only"]["thinking"] <= 0.01, \
        "the paper says tier-only hazard gains nothing on exactly one trace, thinking"
    claim("external unreachable", f"split protocol leaves {min(_un):.1f}--{max(_un):.1f}\\% of the advantage", "", "bailian @ 0.5")
    claim("external hazard, all actions", f"reaches {min(_rt['full'].values()):.1f}--{max(_rt['full'].values()):.1f} times the split-time ceiling with all", "", "bailian hazard")
    claim("external hazard, tier only", f"it reaches {min(_tpos):.1f}--{max(_tpos):.1f} times on three", "", "bailian hazard")

# Table 2 (audit pass 2026-10-19): every cell from a producer.
_ss = load("source_stats.json")
_sm = load("sampling_study.json")
_hb = load("headroom_by_length.json")
if _ss and _sm and _hb:
    print("\nTable 2 (experiments/source_stats.json, sampling_study.json)")
    mc, kv = _ss["mooncake"], _ss["kvct"]["40"]
    claim("Table 2 one-time", f"{100*mc['conversation']['one_time_frac']:.1f} / "
          f"{100*mc['toolagent']['one_time_frac']:.1f}\\% & {100*kv['one_time_frac']:.0f}\\%", "", "source_stats")
    claim("Table 2 mean reuse", f"{mc['conversation']['mean_reuse']:.1f} / {mc['toolagent']['mean_reuse']:.1f} & "
          f"{kv['mean_reuse']:.0f}", "", "source_stats")
    claim("Table 2 blocks/request", f"{mc['conversation']['blocks_per_request']:.0f} / "
          f"{mc['toolagent']['blocks_per_request']:.0f} & $\\sim${round(kv['blocks_per_request'], -2):,.0f}"
          .replace(",", "{,}"), "", "source_stats")
    assert kv["requests"] == 4215
    hs = sorted(r["headroom_mean"] for r in _sm["rows"])
    claim("Table 2 agentic headroom", f"& {hs[0]:.0f}--{hs[-1]:.0f}\\% \\\\", "", "sampling_study means")
    ot = [v["one_time_frac"] * 100 for v in _ss["kvct"].values()]
    claim("agentic one-time range", f"ranges {min(ot):.1f}--{max(ot):.0f}\\% over 20--200", "", "kvct sizes")
    full = {r["trace"]: r["headroom_pct"] for r in _hb["rows"] if r["full_trace"]}
    claim("Table 2 Mooncake headroom pair", f"({full['conversation']:.1f}\\% and {full['toolagent']:.1f}\\%)",
          "", "headroom_by_length")

# Table 3's left column (retracted sorted-prefix runs, the record of the reversal): every cell from
# an artifact. "+87%" at 70 was the classifier variant beside regression values; "-9%" at 100 had no
# shipped artifact (2026-10-19 second pass).
_t3 = {40: load("kvct_placement_fixed.json"), 70: load("kvct_placement_n70.json"),
       100: load("kvct_placement_n100.json"), 150: load("kvct_placement_n150.json")}
if all(_t3.values()):
    print("\nTable 3, sorted-prefix column (kvct_placement_*.json)")
    for n, dd in _t3.items():
        v = [r["regression_headroom_pct"] for r in dd["results"] if r["features"] == "block+session+think"][0]
        claim(f"Table 3 {n} conversations", f"{n} conversations & ${v:+.0f}\\%$", v, "regression, all features")

# The eviction-signal inversion (section V-F): it quoted a superseded artifact until 2026-10-19.
_kv = load("reactive_kvct_sweep.json")
if _kv:
    at5 = [r for r in _kv["rows"] if r["cache_frac"] == 0.05]
    lru = [r["lru"] / r["belady"] for r in at5]
    lfu = [r["h2o"] / r["belady"] for r in at5]
    claim("kvct LRU / LFU at 5%", f"LRU reaches {min(lru):.2f}--{max(lru):.2f} of Belady at a 5\\%\ntier "
          f"while LFU stays near {min(lfu):.2f}--{max(lfu):.2f}", "", "reactive_kvct_sweep @ 5%")

# --- the retracted claims must not reappear ----------------------------------------
RETRACTED = [
    (r"[+0.8\%, +2.9\%]", "the decoupling interval, normalized against an unreachable oracle"),
    ("$\\pm126$ points", "the z-based interval on four draws"),
    ("486--667", "required n computed at an uncorrected alpha, missing the third budget"),
]
for needle, why in RETRACTED:
    if re.sub(r"\s+", " ", needle) in TEX and "earlier revision" not in TEX and \
            "retract" not in TEX.lower():
        failures.append(f"retracted claim {needle!r} reappeared ({why})")

# --- LRB reported as a band ---------------------------------------------------------
# §5.5's mechanism explanation rests on these, and until §AK nothing asserted them -- so the
# paper quoted an artifact that had been moved to superseded/ for being produced by the broken
# implementation. The figures were right for that artifact and wrong for the current one.
d = load("lrb_sweep.json")
if d:
    print("\nLRB censoring sweep (experiments/lrb_sweep.json)")
    def _band(w):
        v = [r["censored_frac"] * 100 for tr in d["traces"].values() for r in tr["rows"]
             if r["memory_window"] == w]
        return min(v), max(v)
    lo, hi = _band(50_000)
    claim("censored share, default window", f"{lo:.0f}--{hi:.0f}\\%", "83--88\\%",
          "censored_frac @ 50k")
    lo, hi = _band(10_000)
    claim("censored share, short window", f"{lo:.0f}--{hi:.0f}\\%", "88--91\\%",
          "censored_frac @ 10k")

d = load("lrb_retraction.json")
if d:
    print("\nLRB band (experiments/lrb_retraction.json)")
    # ARC from the reactive artifact's raw hit rates, the same source as the reactive table:
    # lrb_retraction.json stores ARC/Belady pre-rounded to 4 places, so 0.96253 became 0.9625
    # and printed as 96.2 here and 96.3 there -- one quantity, two values (2026-10-19 pass).
    _rx = load("reactive_real_full_lrb.json")
    _arc = {(t, r["cache_frac"]): r["arc"] / r["belady"]
            for t, v in _rx["traces"].items() for r in v["rows"]}
    for r in d["rows"]:
        lo = min(r["lrb_repair_all_history"], r["lrb_repair_sliding_window"])
        hi = max(r["lrb_repair_all_history"], r["lrb_repair_sliding_window"])
        # The stored ratio was rounded once, the hit rates twice; they may differ by the component
        # rounding (<2e-4), never by more.
        assert abs(_arc[(r["trace"], r["cache_frac"])] - r["arc"]) < 2e-4, "two ARC sources disagree"
        row = f"{100*lo:.1f}--{100*hi:.1f} & {100*_arc[(r['trace'], r['cache_frac'])]:.1f}"
        claim(f"{r['trace']} {r['cache_frac']:.0%} LRB band", row, row, "lrb repairs")
    assert all(x["below_arc_under_all_repairs"] for x in d["rows"]), (
        "LRB is no longer below ARC everywhere; the paper's only surviving LRB claim is void")
    # The band's WIDTH and the reimplementation gap are two different quantities, and a draft
    # compressed them into one wrong number that every row-level check above still passed.
    widest = max(100 * abs(r["lrb_repair_all_history"] - r["lrb_repair_sliding_window"])
                 for r in d["rows"])
    claim("widest disagreement between the two repairs", f"{widest:.1f} points",
          "2.1 points", "lrb_repair_* spread")
    # The "17-point reimplementation gap" assertion that stood here was removed in §AO: its
    # source was a repair produced by an AI agent in an internal review simulation, whose code is
    # not part of this artifact. A number nobody can regenerate is not one the checker should
    # certify, and the paper no longer states it.

# --- capacity ladder, causal magnitude scale (section 5.3, audit H2) ---------------------------
def _sg(v):
    return "0.0" if round(v, 1) == 0 else f"{v:+.1f}"


d = load("capacity_ladder_fixed.json")
if d:
    print("\ncapacity ladder, causal scale (experiments/capacity_ladder_fixed.json)")
    by = {(r["eval"], r["rung"]): r for r in d["results"]}
    for rung in ("constant", "random", "count", "linear", "gbdt", "mlp"):
        c, t = by[("conversation", rung)], by[("toolagent", rung)]
        claim(f"ladder row: {rung}", f"${_sg(c['captured_of_attainable_pct'])}$ & "
              f"${_sg(t['captured_of_attainable_pct'])}$", "", "captured_of_attainable_pct")
    # The paper now claims monotone conversion on BOTH traces, rungs ordered by AUC.
    for tr in ("conversation", "toolagent"):
        rungs = sorted(("count", "linear", "gbdt", "mlp"), key=lambda r: by[(tr, r)]["auc_cross_trace"])
        vals = [by[(tr, r)]["captured_of_attainable_pct"] for r in rungs]
        assert vals == sorted(vals), f"{tr}: ladder not monotone in AUC ({list(zip(rungs, vals))})"
    best = max(by[(tr, r)]["captured_pct"] for tr in ("conversation", "toolagent")
               for r in ("linear", "gbdt", "mlp"))
    assert best < by[("conversation", "ceiling")]["captured_pct"], "a split-time model passed its own ceiling"
    claim("strongest model, of the full gap",
          f"gains {max(by[('conversation', r)]['captured_pct'] for r in ('linear', 'gbdt', 'mlp')):.1f}\\% and "
          f"{max(by[('toolagent', r)]['captured_pct'] for r in ('linear', 'gbdt', 'mlp')):.1f}\\%", "", "ladder")

# --- belief magnitude control, causal scale (section 5.3) -----------------------------------
d = load("belief_controls.json")
if d:
    print("\nbelief controls (experiments/belief_controls.json)")
    for r in d["results"]:
        if r["rung"] == "ceiling":
            continue
        diff = r["magnitude_contribution"]
        dtxt = f"{diff:.1f}" if diff >= 0 else f"${diff:.1f}$"
        row = (f"${_sg(r['native_of_attainable'])}$ & ${_sg(r['common_scale_of_attainable'])}$ & {dtxt}")
        claim(f"{r['trace']} {r['rung']} magnitude row", row, row, "belief control row")
    s2 = d["summary"]
    claim("magnitude contribution range",
          f"{s2['magnitude_min_points']:.0f} to {s2['magnitude_max_points']:.0f} points",
          f"{s2['magnitude_min_points']}-{s2['magnitude_max_points']}", "summary")

# Cost-weight sensitivity (audit M4).
d = load("weight_sensitivity.json")
if d:
    print("\nweight sensitivity (experiments/weight_sensitivity.json)")
    sm = d["summary"]
    claim("unreachable share across weight variants",
          f"{sm['unreachable_min']:.1f}--{sm['unreachable_max']:.1f}\\%", "", "weight variants @ 0.5")
    _rc2 = load("reachable_ceiling.json")
    if _rc2:
        prod = [r["unreachable_pct"] for t in ("conversation", "toolagent")
                for r in _rc2["traces"][t]["rows"]] + [r["unreachable_pct"] for r in d["rows"]]
        claim("abstract: 92--99 across splits and weights",
              f"stays at {min(prod):.0f}--{max(prod):.0f}\\%", "", "splits + weights")

# --- staleness guard ---------------------------------------------------------------
# An artifact older than the code that produces it is not evidence about the current code.
# This is not hypothetical: three artifacts the paper cited were generated before a
# tie-break fix landed in two of the DENOMINATOR routines, and regenerating them moved the
# headline table by up to 39 points. Nothing caught it, because every check compared
# artifacts to the paper and none compared artifacts to their own dependencies.
SRC = ROOT / "src" / "stoa"
from artifact_deps import DEPENDS_ON  # noqa: E402  (shared with record_provenance.py)
# mtime alone gave a false positive during pass 7's mutation testing: restoring a file from
# backup moves the timestamp without changing a byte. The manifest records what each artifact
# was actually generated from, so a moved timestamp with a matching digest is silent and a
# differing digest is reported with both digests named.
_PROV = EXP / ".provenance.json"
_manifest = json.loads(_PROV.read_text()) if _PROV.exists() else None
# The guards that consume DEPENDS_ON run in the final block, after every load(). From here
# they could not see artifacts loaded further down, and one escaped that way: the §AA defect,
# reintroduced by later insertions above them. See §AO.

# The released prose -- README.md, paper/README.md, REPRODUCIBILITY.md -- restates the paper's
# headline numbers, and nothing checked it. Twice now a correction landed in the .tex and not in
# the markdown, most recently leaving the repository's front page advertising the two numbers a
# re-verification pass had just falsified. These are the quantities a reader meets first.
# §5.2's table: the paper's headline, and for twelve passes backed by nothing. Every row is
# now derived from one artifact computed in one frame. See §AN for what the old numbers were
# and why they could not all be right at once.
d = load("reachable_ceiling.json")
if d:
    print("\nreachable ceiling (experiments/reachable_ceiling.json)")
    for _t in ("conversation", "toolagent"):
        rows = d["traces"][_t]["rows"]
        at = [r for r in rows if r["split_frac"] == 0.5][0]
        claim(f"{_t}: blocks with no pre-split access", f"{at['no_pre_split_pct']:.1f}\\%",
              at["no_pre_split_pct"], "no_pre_split_pct @ 0.5")
        claim(f"{_t}: full hindsight gap",
              f"{at['full_hindsight_gap']:,.0f}".replace(",", "{,}"),
              at["full_hindsight_gap"], "prefix_greedy - oracle @ 0.5")
        claim(f"{_t}: attainable gap",
              f"{at['attainable_gap']:,.0f}".replace(",", "{,}"),
              at["attainable_gap"], "prefix_greedy - ceiling @ 0.5")
        claim(f"{_t}: unreachable share", f"{at['unreachable_pct']:.1f}\\%",
              at["unreachable_pct"], "1 - attainable/full @ 0.5")
        lo, hi = d["traces"][_t]["unreachable_min"], d["traces"][_t]["unreachable_max"]
        claim(f"{_t}: unreachable across splits 0.3-0.7",
              f"{lo:.1f}--{hi:.1f}\\%", f"{lo}-{hi}", "split sweep")
    # The pair as the abstract and §5.2 state it, rounded.
    u = [round(d["traces"][t]["rows"][2]["unreachable_pct"], 1)
         for t in ("conversation", "toolagent")]
    claim("unreachable pair as stated in prose", f"{u[0]:.1f}\\% and {u[1]:.1f}\\%",
          f"{u}", "both traces @ 0.5")

# Trace sizes and the length dependence of headroom, from headroom_by_length.json -- which has a
# producer. Its predecessor (mooncake_length_sweep.json) had none, carried no headroom column at
# all, and the paper's "87%" / "81.0% to 87.2%" predated the §W fix while §5.8 said 80% (§AO).
d = load("headroom_by_length.json")
if d:
    print("\nheadroom by length (experiments/headroom_by_length.json)")
    rows = {(r["trace"], r["requests"]): r for r in d["rows"]}
    full = {r["trace"]: r for r in d["rows"] if r["full_trace"]}

    def _tex(n: int) -> str:
        return f"{n:,}".replace(",", "{,}")

    claim("requests per trace, conversation then toolagent",
          f"conversation, {_tex(full['conversation']['requests'])} requests; "
          f"toolagent, {_tex(full['toolagent']['requests'])}",
          f"{full['conversation']['requests']} / {full['toolagent']['requests']}", "full_trace rows")
    for _t, _needle in (("toolagent", "from {a:.1f}\\% to {b:.1f}\\% on toolagent"),
                        ("conversation", "from {a:.1f}\\% to {b:.1f}\\% on conversation")):
        a_, b_ = rows[(_t, 1500)]["headroom_pct"], full[_t]["headroom_pct"]
        claim(f"{_t}: headroom 1,500 requests -> full", _needle.format(a=a_, b=b_),
              f"{a_} -> {b_}", "reference_costs by length")
    _h = round((full["conversation"]["headroom_pct"] + full["toolagent"]["headroom_pct"]) / 2)
    claim("Table 2 headroom at full length", f"heuristic vs.\\ oracle & {_h}\\%",
          f"{full['conversation']['headroom_pct']} / {full['toolagent']['headroom_pct']}",
          "full_trace rows, rounded")

# The synthetic control (§5.2, §5.4): reported, never pooled.
d = load("reachable_ceiling.json")
if d and "synthetic" in d["traces"]:
    print("\nsynthetic control (experiments/reachable_ceiling.json)")
    sy = d["traces"]["synthetic"]
    mid = [r for r in sy["rows"] if r["split_frac"] == 0.5][0]
    claim("synthetic: blocks new at the midpoint", f"{mid['no_pre_split_pct']:.1f}\\%",
          mid["no_pre_split_pct"], "no_pre_split_pct @ 0.5")
    claim("synthetic: unreachable at the midpoint", f"{mid['unreachable_pct']:.1f}\\%",
          mid["unreachable_pct"], "@ 0.5")
    claim("synthetic: unreachable across splits",
          f"{sy['unreachable_min']:.1f}--{sy['unreachable_max']:.1f}\\%",
          f"{sy['unreachable_min']}-{sy['unreachable_max']}", "split sweep")
    claim("synthetic: split instant already reaches", f"{mid['reachable_pct']:.1f}\\%",
          mid["reachable_pct"], "reachable_pct @ 0.5")
    _lo = min(r["no_pre_split_pct"] for r in sy["rows"])
    claim("synthetic: fewest new blocks across splits", f"only {_lo:.0f}\\% of blocks are new",
          _lo, "min no_pre_split_pct")

PROSE_FILES = ("README.md", "paper/README.md", "REPRODUCIBILITY.md")
_arr = load("arrival_admission.json")
_rch = load("reachable_ceiling.json")
if _arr and _rch:
    _by = {(r["trace"], r["arm"]): r for r in _arr["results"]}
    HEADLINE = {}
    for _t in ("conversation", "toolagent"):
        # The causal headline (audit H2): the best split-time policy against plain FCFS admission
        # at arrival, both as a share of the same full hindsight gap.
        _at = [r for r in _rch["traces"][_t]["rows"] if r["split_frac"] == 0.5][0]
        HEADLINE[f"split-time ceiling, {_t}"] = f"{_at['reachable_pct']:.1f}%"
    HEADLINE["estimated prior vs split ceiling"] = "2.3–4.9"
    HEADLINE["tie-convention shift"] = "14–29"
    failures += verify.prose_drift(PROSE_FILES, HEADLINE, ROOT, trigger="arrival")

# The lab notebook legitimately records superseded measurements -- rewriting them would destroy
# the record. But an unmarked one reads as current: §V.2 stated "shuffling the arrival order
# costs 1.8 / 3.0 points, so FCFS ordering contributes little" long after §AC measured 14.1 and
# falsified it. History is fine; unlabelled history is not.
SUPERSEDED_VALUES = {
    "90.4%": "reachable share, conversation (now 83.5%)",
    "52.8%": "reachable share, toolagent (now 19.0%)",
    "89.5%": "FCFS of attainable, toolagent (now 50.2%)",
    "93.2%": "FCFS of attainable, conversation (now 87.4%)",
    "-825.7%": "ladder constant, toolagent (now -60.6%)",
    "\u2212825.7%": "ladder constant, toolagent (now -60.6%)",
}
# "paper carried | regenerated" is an explicit before/after column header -- the value is
# labelled superseded, just in different words. Accepting it is not a weakening.
# The 68 checks above are PRESENCE tests: they ask whether the correct value appears in the
# paper. They do not ask whether a superseded one also appears. With 90% and 53% injected into
# the abstract -- the two values §AC falsified -- all 68 passed, because the correct values were
# still there in §5. A retracted number can therefore sit in the abstract indefinitely.
# The paper does cite superseded figures legitimately, but only in the paragraph that retracts
# them, so the marker scope is the paragraph rather than the section.
SUPERSEDED_IN_PAPER = {
    "90.4": "reachable share, conversation (now 83.5)",
    "52.8": "reachable share, toolagent (now 19.0)",
    "89.5": "FCFS of attainable, toolagent (now 50.2)",
    "93.2": "FCFS of attainable, conversation (now 87.4)",
    "-154.6": "ladder constant, conversation (now -46.1)",
    "-825.7": "ladder constant, toolagent (now -60.6)",
    "64--88": "calibration headroom (now 60--83)",
    "67--90": "calibration headroom under 256-token blocks (now 60--83, §AO)",
    "19 points": "LRB band width (2.1 between repairs, 17 to the reimplementation)",
    # Non-bold, so the emphasis guard could not see them; they survived the §AN correction.
    "share from 3\\%": "split-instant reachable share (now 5.3%)",
    "and 2\\% to": "split-instant reachable share, toolagent (now 4.1%)",
    "87.2\\%": "headroom at full length (now 79.8-79.9%)",
    "The 97\\%": "the unreachable share (now ~95%, section 5.2)",
    # Retracted CLAIMS, not numbers (audit H6). The conclusion and the related-work section both
    # restated the decoupling result as "the contribution" for months: no number, so no number
    # check could see them. A retracted claim may appear only inside a paragraph retracting it.
    "decoupling": "the retracted decoupling result (section T)",
    "neither yes nor no": "the retracted decoupling framing",
    "ranking quality improves substantially while": "the retracted decoupling claim",
    "an oracle's ordering would recover": "contradicts section 5.2: most of the gap is unreachable",
    "ranking does not convert": "the retracted decoupling claim",
    # The leaky arrival design's readings (audit H2, notebook section AP).
    "50.2": "FCFS of attainable, toolagent, from the leaky arrival design",
    "19.0\\%": "reachable share at arrival, toolagent, from the leaky arrival design",
    "worth 14 points": "arrival order's value, from the leaky arrival design",
    "timing multiplier": "arrival ceiling / split ceiling, from the leaky arrival design",
    "of what is reachable": "FCFS against a mixed-scale ceiling that is not an upper bound",
    # The unfair order comparison (notebook AQ) and residue found on the 2026-10-19 pass.
    "2--26": "FCFS lead over the ranker, a magnitude difference read as order",
    "10--27": "FCFS lead over a shuffled order, a magnitude difference read as order",
    "81--94": "agentic headroom from nested-prefix samples",
    "0.88--0.95": "kvct LRU/Belady from a superseded artifact",
    "deciding well": "order barely matters; the withdrawn FCFS reading",
    "eleventh entry": "a catalogue entry that no longer exists",
    # Values from billing creations as reads (notebook AS).
    "94.7": "unreachable share with creations billed (now 84.4)",
    "95.9": "unreachable share with creations billed (now 87.6)",
    "18--39": "tie shift under the creation-billing defect (now 14--29)",
    "at least three times": "a claim the corrected bill falsifies",
    "within 2.1 points of each other": "order spread under the creation-billing defect (now 5.3)",
}
for _tex in [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))]:
    failures += verify.unmarked_superseded(
        _tex.read_text(), SUPERSEDED_IN_PAPER,
        label=str(_tex.relative_to(ROOT)), markers=verify.PAPER_MARKERS, scope="paragraph")

# A dangling path carries no wrong number, so every earlier stale-prose sweep missed it.
# Makefile included: it is now the committee's entry point, so a target naming a deleted
# script is the same defect as a claim table row pointing at a moved artifact (§AK).
for _doc in ("REPRODUCIBILITY.md", "README.md", "paper/README.md", "data/README.md",
             "Makefile"):
    _p = ROOT / _doc
    if _p.exists():
        _t = _p.read_text()
        failures += verify.dangling_paths(_t, ROOT, _doc)
        failures += verify.broken_entry_points(_t, _doc)

# Stated test counts (audit M1): the README said 173, REPRODUCIBILITY 163 and the Makefile 231 while
# the suite held 261. A count is a claim like any other; collect the suite and compare.
import re as _re
import subprocess as _sp

_col = _sp.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
               capture_output=True, text=True)
_m = _re.search(r"(\d+) tests? collected", _col.stdout)
if _m:
    _n = int(_m.group(1))
    _pat = _re.compile(r"(?:# |\| |all \*\*|— |min   )(\d+)\*{0,2} tests\b")
    for _doc in ("README.md", "REPRODUCIBILITY.md", "Makefile", "paper/README.md"):
        _p = ROOT / _doc
        if not _p.exists():
            continue
        for _hit in _pat.finditer(_p.read_text()):
            if int(_hit.group(1)) != _n:
                failures.append(f"{_doc} says {_hit.group(1)} tests; the suite collects {_n}.")
else:
    failures.append("could not collect the test suite to check stated test counts")

NOTEBOOK = ROOT / "docs" / "claims_dependency.md"
if NOTEBOOK.exists():
    failures += verify.unmarked_superseded(
        NOTEBOOK.read_text(), SUPERSEDED_VALUES, label="docs/claims_dependency.md")

# Figures are artifacts too, one level down: make_figures.py reads experiments/*.json and
# writes paper/figs/*.pdf. The paper embeds the PDF, not the JSON, so a figure older than the
# data it plots ships a picture of a number the text no longer makes -- and nothing said so.
# The build that caught this had figures fifteen days older than the artifacts they drew from.
# Only what the paper embeds. Five rules lived here for two passes; four aimed at figures no
# `.tex` file includes, so the guard read as covering the artifact -> figure edge while
# covering one fifth of it. `make_figures.py` still produces the others for a longer version;
# they are regenerable and unguarded, which is the honest description of an unused plot.
FIG_SOURCES = {
    "fig_reactive.pdf": ("reactive_real_full_lrb.json",),
}
FIGS = ROOT / "paper" / "figs"
failures += verify.stale_figures(FIG_SOURCES, FIGS, EXP)
# A freshness rule can only protect what it is told about, and for two passes the list did
# not match what the paper embeds. `fig_architecture` (a schematic, no longer embedded since §IV was
# compressed) stays declared data-free, because an exemption on the record is not an omission.
failures += verify.figure_coverage_gaps(
    [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))],
    FIG_SOURCES, data_free=("fig_architecture.pdf",))

# The last edge of the chain: a correction in the .tex is not a correction until the PDF is
# rebuilt, and the PDF is what a reviewer reads.
_paper = ROOT / "paper"
# The constrained quantity is the BODY's page count, not the file's. Measuring the file
# agreed with the truth only while the references happened to fit on the body's last page.
# The one requirement the CfP says can cost the paper, and the one no guard had ever looked
# at: `ANONYMIZED` sat in main.tex through eleven passes. Network check is opt-in
# (STOA_CHECK_URL=1) so an offline run cannot silently report it as passing.
failures += verify.availability_url_problems(
    _paper / "main.tex", check_network=os.environ.get("STOA_CHECK_URL") == "1")
# ICDE 2027: "12 pages, excluding references and the AI-generated content acknowledgement".
# The acknowledgement precedes the references, so the body ends at whichever comes first.
failures += verify.page_limit_violation(_paper / "main.pdf", limit=12,
                                        end_headings=("ACKNOWLEDGMENT", "ACKNOWLEDGMENTS",
                                                      "REFERENCES"))
failures += verify.bibtex_problems(_paper / "main.blg")
failures += verify.author_block_problems(_paper / "main.tex")
failures += verify.leaked_annotations(_paper / "main.pdf")
failures += verify.type3_fonts(_paper / "main.pdf")
_cited = set()
for _t in [_paper / "main.tex", *sorted((_paper / "sections").glob("*.tex"))]:
    for _m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", _t.read_text()):
        _cited |= {k.strip() for k in _m.split(",")}
failures += verify.citation_hygiene((ROOT / "docs" / "references.bib").read_text(), _cited)
PDF_SOURCES = [_paper / "main.tex", *sorted((_paper / "sections").glob("*.tex")),
               *[FIGS / f for f in FIG_SOURCES], ROOT / "docs" / "references.bib"]
_pdf_manifest = _paper / ".build_manifest.json"
failures += verify.stale_paper_pdf(
    _paper / "main.pdf", PDF_SOURCES,
    json.loads(_pdf_manifest.read_text()) if _pdf_manifest.exists() else None)

# Which emphasised numbers is nothing watching? MIN_CHECKS below guarantees the checker does a
# certain AMOUNT of work; this guarantees it works on the right things. Twelve passes guarded
# numbers that were in artifacts and never asked the reverse question -- which is how the
# paper's headline 97% sat unbacked in §5.2.
failures += verify.unbacked_emphasised_numbers(
    [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))],
    " ".join(_ASSERTED),
    allow=(
        # Definitional, not measured: these come from the formulation or the venue, not a run.
        "12 pages", "50", "fifty",
    ))

# --- artifact guards, LAST so every load() above is visible to them -------------------
# Artifacts this paper does not cite are not held to freshness here: their staleness cannot
# affect a number the paper states. They are named, not silently dropped.
_IN_PAPER = {k: v for k, v in DEPENDS_ON.items() if k not in NOT_IN_THIS_PAPER}
print(f"\n[skip] freshness of {sorted(set(DEPENDS_ON) - set(_IN_PAPER))}: not cited by this paper")
failures += verify.undeclared_artifacts(_LOADED, DEPENDS_ON, EXP)
failures += verify.stale_artifacts(_IN_PAPER, EXP, SRC, manifest=_manifest)
failures += verify.producer_problems(
    _IN_PAPER, ROOT / "scripts", SRC,
    readers=("check_paper_numbers.py", "make_figures.py", "record_provenance.py"),
    producers={"reactive_real_full_lrb.json": "run_reactive_real.py",
               "capacity_ladder_fixed.json": "run_capacity_ladder.py"})

# --- coverage guards, LAST so `checked` is final -----------------------------------
# Sitting mid-file, this counted only the checks declared above it and passed while
# a third of the paper went unchecked.
# Every artifact the ICDE paper cites, and only those. `capacity_ladder.json` (superseded) and
# `split_sensitivity.json` (not cited) used to be here, so a reproducer was told they were required.
REQUIRED = ("reactive_real_full_lrb.json", "headroom_by_length.json", "lrb_retraction.json",
            "reachable_ceiling.json", "lrb_sweep.json", "calibration_sensitivity.json",
            "sampling_study.json", "arrival_admission.json", "arrival_diagnostics.json",
            "belief_controls.json", "capacity_ladder_fixed.json", "weight_sensitivity.json",
            "source_stats.json", "reactive_kvct_sweep.json", "tier_only.json",
            "hazard_prior.json", "hazard_prior_tier.json", "equal_horizon.json", "reachable_by_length.json",
            "hazard_prior_rescaled.json", "hazard_prior_tier_rescaled.json", "online_admission.json",
            "arrival_admission_reuse.json", "tier_only_reuse.json",
            "bailian_reachable_birthx.json", "bailian_hazard_birthx.json", "reactive_cost.json")
missing = [n for n in REQUIRED if not (EXP / n).exists()]
if missing:
    failures.append("artifacts absent, so their claims went unchecked: " + ", ".join(missing))
MIN_CHECKS = 60
if checked < MIN_CHECKS:
    failures.append(f"only {checked} claims checked; expected at least {MIN_CHECKS}. "
                    "A pass with too few checks means artifacts are missing, not that the "
                    "paper is correct. Raise this floor whenever you add claims, and lower "
                    "it only when you delete them deliberately.")

# "located" was wrong: this is checks PASSED, not claims found. The earlier wording let a
# run with three failures read as a coverage shortfall, which is a different and milder thing.
print(f"\n{checked - len(failures)}/{checked} checks passed "
      f"({len(failures)} failing).")
if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checked numbers appear verbatim in paper/*.tex.")
