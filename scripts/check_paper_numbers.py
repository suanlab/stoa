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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
import re

_RAW = " ".join(p.read_text() for p in
                [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))])
# Collapse runs of whitespace. LaTeX table columns are padded for readability in the source,
# so an otherwise-correct needle fails on cosmetic alignment -- which teaches you to ignore
# the checker. Whitespace is the one difference that never changes a number.
TEX = re.sub(r"\s+", " ", _RAW)

failures: list[str] = []
checked = 0


def load(name):
    p = EXP / name
    return json.loads(p.read_text()) if p.exists() else None


def claim(label: str, needle: str, derived, source: str) -> None:
    """Assert `needle` appears in the LaTeX source, and report what the artifact says.

    `derived` is the value recomputed from the artifact, printed alongside so a mismatch is
    diagnosable rather than just red.
    """
    global checked
    checked += 1
    ok = re.sub(r"\s+", " ", needle) in TEX
    print(f"  [{'ok ' if ok else 'MISS'}] {label:<52} {derived}   ({source})")
    if not ok:
        failures.append(f"{label}: expected {needle!r} in the paper source, from {source}")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


print("Checking paper claims against experiments/*.json\n")

# --- reactive tiering + ARC + LRB -------------------------------------------------
d = load("reactive_real_full_lrb.json") or load("reactive_real_full.json")
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

# --- arrival-time admission: the paper's new headline ------------------------------
d = load("arrival_admission.json")
if d:
    print("\narrival admission (experiments/arrival_admission.json)")
    by = {(r["trace"], r["arm"]): r for r in d["results"]}
    for trace in ("conversation", "toolagent"):
        for arm, label in (("arrival-learned", "learned"), ("arrival-fcfs", "FCFS"),
                           ("arrival-shuffled", "shuffled"), ("arrival-ceiling", "ceiling")):
            r = by.get((trace, arm))
            if not r:
                continue
            # Accept the cells with or without \textbf: emphasis is typography, and a
            # checker that fails on it is a checker people learn to ignore.
            g, a_ = r["captured_pct"], r["captured_of_attainable_pct"]
            plain = f"{g:.1f}\\% & {a_:.1f}\\%"
            bold = f"\\textbf{{{g:.1f}\\%}} & \\textbf{{{a_:.1f}\\%}}"
            found = any(re.sub(r"\s+", " ", n) in TEX for n in (plain, bold))
            # Report through claim() so the count and failure list stay in one place; pass
            # whichever form is present so the needle check cannot disagree with `found`.
            claim(f"{trace} {label} row", bold if found and plain not in TEX else plain,
                  plain, "arrival row")
        ceil = by[(trace, "arrival-ceiling")]["captured_pct"]
        claim(f"{trace} reachable share at arrival", f"{ceil:.1f}\\%",
              f"{ceil:.1f}%", "arrival-ceiling")

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
d = load("lrb_retraction.json")
if d:
    print("\nLRB band (experiments/lrb_retraction.json)")
    for r in d["rows"]:
        lo = min(r["lrb_repair_all_history"], r["lrb_repair_sliding_window"])
        hi = max(r["lrb_repair_all_history"], r["lrb_repair_sliding_window"])
        row = f"{100*lo:.1f}--{100*hi:.1f} & {100*r['arc']:.1f}"
        claim(f"{r['trace']} {r['cache_frac']:.0%} LRB band", row, row, "lrb repairs")
    assert all(x["below_arc_under_all_repairs"] for x in d["rows"]), (
        "LRB is no longer below ARC everywhere; the paper's only surviving LRB claim is void")

# --- capacity ladder on the corrected metric ---------------------------------------
d = load("capacity_ladder_fixed.json")
if d:
    print("\ncapacity ladder, corrected metric (experiments/capacity_ladder_fixed.json)")
    by = {(r["eval"], r["rung"]): r for r in d["results"]}
    for rung in ("constant", "random", "count", "linear", "gbdt", "mlp"):
        c = by.get(("conversation", rung))
        t = by.get(("toolagent", rung))
        if not (c and t):
            continue
        # Both cells of the row, so a lone number cannot satisfy the check.
        row = (f"${c['captured_of_attainable_pct']:.1f}$ & "
               f"${t['captured_of_attainable_pct']:.1f}$")
        claim(f"ladder row: {rung}", row, row, "captured_of_attainable_pct")
    ordered = [by[("conversation", r)]["captured_of_attainable_pct"]
               for r in ("constant", "count", "linear", "mlp") if ("conversation", r) in by]
    assert ordered == sorted(ordered), (
        "the ladder is no longer monotone in AUC; the paper's claim depends on it")

# --- belief magnitude control ------------------------------------------------------
d = load("belief_controls.json")
if d:
    print("\nbelief controls (experiments/belief_controls.json)")
    for r in d["results"]:
        if r["rung"] == "ceiling":
            continue
        row = (f"${r['native_of_attainable']:+.1f}$ & ${r['common_scale_of_attainable']:+.1f}$ & "
               f"{abs(r['magnitude_contribution']):.1f}")
        claim(f"{r['trace']} {r['rung']} magnitude row", row, row, "belief control row")
    s2 = d["summary"]
    claim("magnitude contribution range",
          f"{s2['magnitude_min_points']:.1f} to {s2['magnitude_max_points']:.1f} points",
          f"{s2['magnitude_min_points']:.1f}-{s2['magnitude_max_points']:.1f}", "summary")


# --- coverage guards, LAST so `checked` is final -----------------------------------
# Sitting mid-file, this counted only the checks declared above it and passed while
# a third of the paper went unchecked.
REQUIRED = ("reactive_real_full_lrb.json", "capacity_ladder.json",
            "mooncake_length_sweep.json", "mooncake_attribution_sweep.json",
            "split_sensitivity.json", "lrb_retraction.json", "locomo_power.json",
            "eval_locomo_powered.json", "representation_axis.json",
            "calibration_sensitivity.json", "sampling_study.json",
            "arrival_admission.json", "belief_controls.json",
            "capacity_ladder_fixed.json")
missing = [n for n in REQUIRED if not (EXP / n).exists()]
if missing:
    failures.append("artifacts absent, so their claims went unchecked: " + ", ".join(missing))
MIN_CHECKS = 60
if checked < MIN_CHECKS:
    failures.append(f"only {checked} claims checked; expected at least {MIN_CHECKS}. "
                    "A pass with too few checks means artifacts are missing, not that the "
                    "paper is correct. Raise this floor whenever you add claims, and lower "
                    "it only when you delete them deliberately.")

print(f"\n{checked - len(failures)}/{checked} claims located in the paper source.")
if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checked numbers appear verbatim in paper/*.tex.")
