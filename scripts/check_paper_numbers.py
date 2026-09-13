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


def load(name):
    """Read an artifact, recording the name so the staleness map can be audited against it."""
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
    for r in d["rows"]:
        lo = min(r["lrb_repair_all_history"], r["lrb_repair_sliding_window"])
        hi = max(r["lrb_repair_all_history"], r["lrb_repair_sliding_window"])
        row = f"{100*lo:.1f}--{100*hi:.1f} & {100*r['arc']:.1f}"
        claim(f"{r['trace']} {r['cache_frac']:.0%} LRB band", row, row, "lrb repairs")
    assert all(x["below_arc_under_all_repairs"] for x in d["rows"]), (
        "LRB is no longer below ARC everywhere; the paper's only surviving LRB claim is void")
    # The band's WIDTH and the reimplementation gap are two different quantities, and a draft
    # compressed them into one wrong number that every row-level check above still passed.
    widest = max(100 * abs(r["lrb_repair_all_history"] - r["lrb_repair_sliding_window"])
                 for r in d["rows"])
    claim("widest disagreement between the two repairs", f"{widest:.1f} points",
          "2.1 points", "lrb_repair_* spread")
    note = d.get("independent_reimplementation", "")
    # Targeted, not greedy: an earlier version matched the "/10%" of "conversation/10%"
    # and reported a 53-point gap. The two readings are the reviewer's and ours.
    m = re.search(r"~?(\d+(?:\.\d+)?)% at .*? where ours gives (\d+(?:\.\d+)?)%", note)
    if m:
        gap = abs(float(m.group(1)) - float(m.group(2)))
        claim("independent reimplementation gap", f"{gap:.0f}-point difference",
              "17-point difference", "independent_reimplementation")

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
    # The paper claims monotonicity on toolagent ONLY, and says so explicitly. Asserting it
    # on both traces is what an earlier revision did, from numbers predating the baseline
    # tie-break fix.
    tool = [by[("toolagent", r)]["captured_of_attainable_pct"]
            for r in ("constant", "count", "linear", "mlp") if ("toolagent", r) in by]
    assert tool == sorted(tool), (
        f"toolagent ladder is no longer monotone in AUC: {tool}; the paper claims it is")
    conv = [by[("conversation", r)]["captured_of_attainable_pct"]
            for r in ("constant", "count", "linear", "mlp") if ("conversation", r) in by]
    assert conv != sorted(conv), (
        f"conversation ladder became monotone: {conv}; the paper says it is not, so update it")

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


# --- staleness guard ---------------------------------------------------------------
# An artifact older than the code that produces it is not evidence about the current code.
# This is not hypothetical: three artifacts the paper cited were generated before a
# tie-break fix landed in two of the DENOMINATOR routines, and regenerating them moved the
# headline table by up to 39 points. Nothing caught it, because every check compared
# artifacts to the paper and none compared artifacts to their own dependencies.
SRC = ROOT / "src" / "stoa"
DEPENDS_ON = {
    # artifact -> modules whose behaviour it encodes
    "arrival_admission.json": ("sequential.py", "mooncake.py"),
    "capacity_ladder_fixed.json": ("sequential.py", "gbdt.py", "mooncake.py"),
    "calibration_sensitivity.json": ("sequential.py", "calibration.py", "simulator.py"),
    "belief_controls.json": ("sequential.py",),
    "reactive_real_full_lrb.json": ("eval/online.py", "experts.py", "lrb.py"),
    "lrb_retraction.json": ("lrb.py",),
    "eval_locomo_powered.json": ("eval/memqa.py", "stats.py"),
    "representation_axis.json": ("eval/representation.py", "stats.py"),
    "locomo_power.json": ("stats.py",),
    "lrb_sweep.json": ("lrb.py", "eval/online.py"),
    "sampling_study.json": ("kvct.py",),
}
# The guard's own failure mode: add an artifact, forget to declare its dependencies, and it
# is exempt from staleness checking forever without anything saying so.
failures += verify.undeclared_artifacts(_LOADED, DEPENDS_ON, EXP)
# mtime alone gave a false positive during pass 7's mutation testing: restoring a file from
# backup moves the timestamp without changing a byte. The manifest records what each artifact
# was actually generated from, so a moved timestamp with a matching digest is silent and a
# differing digest is reported with both digests named.
_PROV = EXP / ".provenance.json"
_manifest = json.loads(_PROV.read_text()) if _PROV.exists() else None
failures += verify.stale_artifacts(DEPENDS_ON, EXP, SRC, manifest=_manifest)

# The released prose -- README.md, paper/README.md, REPRODUCIBILITY.md -- restates the paper's
# headline numbers, and nothing checked it. Twice now a correction landed in the .tex and not in
# the markdown, most recently leaving the repository's front page advertising the two numbers a
# re-verification pass had just falsified. These are the quantities a reader meets first.
# The cost of shuffling the arrival order: the claim §AC falsified and the paper now states
# in corrected form. Emphasised in both the introduction and §5.4 and asserted by nothing.
# Trace sizes: stated in the paper as a bare pair, in the reverse of the column order every
# other table uses, and asserted by nothing. The artifact labels its rows, so the mapping is
# recoverable rather than a matter of which one a reader assumes comes first.
d = load("mooncake_length_sweep.json")
if d:
    print("\ntrace sizes (experiments/mooncake_length_sweep.json)")
    full = {r["trace"]: r["requests"] for r in d["grid"] if r.get("full_trace")}
    if {"conversation", "toolagent"} <= set(full):
        def _tex(n: int) -> str:
            # LaTeX thousands separator, applied to the NUMBER only -- a blanket
            # str.replace(",", "{,}") also rewrites the prose commas around it.
            return f"{n:,}".replace(",", "{,}")

        claim("requests per trace, conversation then toolagent",
              f"conversation, {_tex(full['conversation'])} requests; "
              f"toolagent, {_tex(full['toolagent'])}",
              f"{full['conversation']} / {full['toolagent']}", "full_trace rows")

_arr0 = load("arrival_admission.json")
if _arr0:
    _b = {(r["trace"], r["arm"]): r for r in _arr0["results"]}
    print("\narrival order (experiments/arrival_admission.json)")
    for _t, _needle in (("conversation", "3.3 points"), ("toolagent", "14.1 points")):
        _cost = (_b[(_t, "arrival-fcfs")]["captured_of_attainable_pct"]
                 - _b[(_t, "arrival-shuffled")]["captured_of_attainable_pct"])
        claim(f"{_t}: cost of shuffling the arrival order", f"{_cost:.1f} points",
              f"{_cost:.2f}", "fcfs - shuffled, of attainable")

PROSE_FILES = ("README.md", "paper/README.md", "REPRODUCIBILITY.md")
_arr = load("arrival_admission.json")
if _arr:
    _by = {(r["trace"], r["arm"]): r for r in _arr["results"]}
    HEADLINE = {}
    for _t in ("conversation", "toolagent"):
        # The reachable share IS the ceiling arm's captured_pct: what a clairvoyant policy
        # restricted to blocks it could have seen actually captures.
        HEADLINE[f"reachable share, {_t}"] = f"{_by[(_t, 'arrival-ceiling')]['captured_pct']:.1f}"
        HEADLINE[f"FCFS of attainable, {_t}"] = (
            f"{_by[(_t, 'arrival-fcfs')]['captured_of_attainable_pct']:.1f}")
    failures += verify.prose_drift(PROSE_FILES, HEADLINE, ROOT, trigger="reachable share")

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
    "64--88": "calibration headroom (now 67--90)",
    "19 points": "LRB band width (2.1 between repairs, 17 to the reimplementation)",
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
    "fig_reactive.pdf": ("reactive_real_full_lrb.json", "reactive_real_full.json"),
}
FIGS = ROOT / "paper" / "figs"
failures += verify.stale_figures(FIG_SOURCES, FIGS, EXP)
# A freshness rule can only protect what it is told about, and for two passes the list did
# not match what the paper embeds. `fig_architecture` is a schematic: exempt from staleness,
# but declared so, because an exemption on the record is not the same as an omission.
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
failures += verify.page_limit_violation(_paper / "main.pdf", limit=12)
failures += verify.stale_paper_pdf(
    _paper / "main.pdf",
    [_paper / "main.tex", *sorted((_paper / "sections").glob("*.tex")),
     *[FIGS / f for f in FIG_SOURCES], ROOT / "docs" / "references.bib"])

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
