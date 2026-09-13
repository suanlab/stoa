"""Tests for the freshness guards in `stoa.verify`.

Every guard here was written after a defect reached a draft, and each is tested in both
directions: it fires on the violation it exists for, and it stays silent when the tree is
clean. The negative half matters as much as the positive one -- for four re-verification
passes these guards lived inline in a script with no tests at all, so a refactor could have
deleted any of them and every pass would still have reported clean.

The order of business in each test is the order the defect actually happened: build a tree
that is correct, confirm silence, then age one file and confirm the complaint.
"""

import time

import pytest

from stoa import verify


def _write(p, text="x", age=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if age is not None:
        import os
        os.utime(p, (age, age))
    return p


NOW = 1_000_000.0
OLD = NOW - 86_400  # a day behind


# --- stale_artifacts: src/*.py -> experiments/*.json ------------------------------------

def test_stale_artifact_detected_when_module_is_newer(tmp_path):
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(exp / "a.json", age=OLD)
    _write(src / "sequential.py", age=NOW)
    out = verify.stale_artifacts({"a.json": ("sequential.py",)}, exp, src)
    assert len(out) == 1 and "STALE ARTIFACT" in out[0] and "a.json" in out[0]


def test_fresh_artifact_is_silent(tmp_path):
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(src / "sequential.py", age=OLD)
    _write(exp / "a.json", age=NOW)
    assert verify.stale_artifacts({"a.json": ("sequential.py",)}, exp, src) == []


def test_every_declared_module_is_checked_not_just_the_first(tmp_path):
    """The real artifact declares up to three modules; a loop that stopped at the first
    would have passed while the third was newer."""
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(exp / "a.json", age=NOW - 100)
    _write(src / "one.py", age=OLD)
    _write(src / "two.py", age=OLD)
    _write(src / "three.py", age=NOW)
    out = verify.stale_artifacts({"a.json": ("one.py", "two.py", "three.py")}, exp, src)
    assert len(out) == 1 and "three.py" in out[0]


def test_missing_artifact_is_not_reported_as_stale(tmp_path):
    """Absence is a different failure, reported by the REQUIRED guard, not this one."""
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(src / "sequential.py", age=NOW)
    exp.mkdir(parents=True, exist_ok=True)
    assert verify.stale_artifacts({"a.json": ("sequential.py",)}, exp, src) == []


# --- undeclared_artifacts: the guard's own failure mode ----------------------------------

def test_artifact_read_but_undeclared_is_flagged(tmp_path):
    exp = tmp_path / "experiments"
    _write(exp / "declared.json")
    _write(exp / "forgotten.json")
    out = verify.undeclared_artifacts(
        ["declared.json", "forgotten.json"], {"declared.json": ("m.py",)}, exp)
    assert len(out) == 1 and "forgotten.json" in out[0] and "declared.json" not in out[0]


def test_undeclared_but_absent_artifact_is_not_flagged(tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir(parents=True)
    assert verify.undeclared_artifacts(["never_written.json"], {}, exp) == []


# --- stale_figures: experiments/*.json -> paper/figs/*.pdf -------------------------------

def test_figure_older_than_its_data_is_flagged(tmp_path):
    figs, exp = tmp_path / "figs", tmp_path / "experiments"
    _write(figs / "f.pdf", age=OLD)
    _write(exp / "d.json", age=NOW)
    out = verify.stale_figures({"f.pdf": ("d.json",)}, figs, exp)
    assert len(out) == 1 and "STALE FIGURE" in out[0]


def test_figure_flagged_once_even_with_several_stale_sources(tmp_path):
    figs, exp = tmp_path / "figs", tmp_path / "experiments"
    _write(figs / "f.pdf", age=OLD)
    _write(exp / "a.json", age=NOW)
    _write(exp / "b.json", age=NOW)
    assert len(verify.stale_figures({"f.pdf": ("a.json", "b.json")}, figs, exp)) == 1


def test_fresh_figure_is_silent(tmp_path):
    figs, exp = tmp_path / "figs", tmp_path / "experiments"
    _write(exp / "d.json", age=OLD)
    _write(figs / "f.pdf", age=NOW)
    assert verify.stale_figures({"f.pdf": ("d.json",)}, figs, exp) == []


# --- stale_paper_pdf: sections/*.tex -> main.pdf -----------------------------------------

def test_pdf_older_than_a_section_is_flagged(tmp_path):
    pdf = _write(tmp_path / "main.pdf", age=OLD)
    tex = _write(tmp_path / "s.tex", age=NOW)
    out = verify.stale_paper_pdf(pdf, [tex])
    assert len(out) == 1 and "STALE PDF" in out[0]


def test_pdf_names_every_source_it_is_behind(tmp_path):
    pdf = _write(tmp_path / "main.pdf", age=OLD)
    a = _write(tmp_path / "a.tex", age=NOW)
    b = _write(tmp_path / "b.tex", age=NOW)
    out = verify.stale_paper_pdf(pdf, [a, b])
    assert len(out) == 1 and "a.tex" in out[0] and "b.tex" in out[0]


def test_fresh_pdf_is_silent(tmp_path):
    tex = _write(tmp_path / "s.tex", age=OLD)
    pdf = _write(tmp_path / "main.pdf", age=NOW)
    assert verify.stale_paper_pdf(pdf, [tex]) == []


# --- prose_drift: released assertions ----------------------------------------------------

HEAD = {"reachable share, conversation": "83.5", "FCFS, toolagent": "50.2"}


def test_prose_missing_a_current_value_is_flagged(tmp_path):
    _write(tmp_path / "README.md", "reachable share rises to 90.0% and FCFS reaches 50.2%")
    out = verify.prose_drift(["README.md"], HEAD, tmp_path, trigger="reachable share")
    assert len(out) == 1 and "83.5" in out[0]


def test_prose_with_all_current_values_is_silent(tmp_path):
    _write(tmp_path / "README.md", "reachable share rises to 83.5% and FCFS reaches 50.2%")
    assert verify.prose_drift(["README.md"], HEAD, tmp_path, trigger="reachable share") == []


def test_file_that_does_not_restate_the_result_is_not_held_to_it(tmp_path):
    """A README about something else must not be forced to quote numbers it never mentions."""
    _write(tmp_path / "README.md", "This directory holds the raw traces.")
    assert verify.prose_drift(["README.md"], HEAD, tmp_path, trigger="reachable share") == []


def test_absent_prose_file_is_not_an_error(tmp_path):
    assert verify.prose_drift(["nope.md"], HEAD, tmp_path, trigger="reachable share") == []


# --- unmarked_superseded: released records -----------------------------------------------

VALS = {"90.4%": "reachable share, conversation (now 83.5%)"}


def test_superseded_value_without_a_marker_is_flagged():
    text = "## V.2 The arrival experiment\n\nDeciding on arrival attains 90.4% of the gap.\n"
    out = verify.unmarked_superseded(text, VALS)
    assert len(out) == 1 and "90.4%" in out[0]


def test_banner_in_the_same_section_silences_it():
    text = ("## V.2 The arrival experiment\n\n"
            "> **SUPERSEDED by AC.** Current value is 83.5%.\n\n"
            "Deciding on arrival attains 90.4% of the gap.\n")
    assert verify.unmarked_superseded(text, VALS) == []


def test_marker_matching_is_case_insensitive():
    """Prose says 'the claim AC falsified'; headers shout SUPERSEDED. Both are markers."""
    text = "## V.2\n\nThis is the claim AC falsified: it attains 90.4%.\n"
    assert verify.unmarked_superseded(text, VALS) == []


def test_banner_in_a_previous_section_does_not_silence_the_next():
    """The defect this guard exists for: a correction 150 lines below the claim it corrects.
    A marker must be in the value's own section, not merely somewhere in the file."""
    text = ("## U Something\n\n> **SUPERSEDED.**\n\n"
            "## V.2 The arrival experiment\n\nAttains 90.4% of the gap.\n")
    out = verify.unmarked_superseded(text, VALS)
    assert len(out) == 1


def test_marker_after_the_value_does_not_silence_it():
    """A banner placed below the claim it corrects is what the notebook already had."""
    text = "## V.2\n\nAttains 90.4% of the gap.\n\n> **SUPERSEDED by AC.**\n"
    assert len(verify.unmarked_superseded(text, VALS)) == 1


def test_before_and_after_column_counts_as_marked():
    text = "## AC.1 What moved\n\n| quantity | paper carried | regenerated |\n| x | 90.4% | 83.5% |\n"
    assert verify.unmarked_superseded(text, VALS) == []


def test_line_number_reported_is_one_indexed():
    text = "## V.2\nfiller\nAttains 90.4%.\n"
    out = verify.unmarked_superseded(text, VALS)
    assert out[0].startswith("notebook:3 ")


def test_label_is_used_in_the_message():
    text = "## V.2\nAttains 90.4%.\n"
    out = verify.unmarked_superseded(text, VALS, label="docs/x.md")
    assert out[0].startswith("docs/x.md:2 ")


# --- page limit: the constrained quantity is the body, not the file ----------------------

def test_page_limit_check_refuses_to_pass_when_it_cannot_run(tmp_path, monkeypatch):
    """A page-limit check that cannot run must not look like one that passed."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    pdf = _write(tmp_path / "main.pdf")
    out = verify.page_limit_violation(pdf)
    assert len(out) == 1 and "UNCHECKED" in out[0]


def test_page_limit_reports_missing_pdf_rather_than_passing(tmp_path):
    out = verify.page_limit_violation(tmp_path / "absent.pdf")
    assert len(out) == 1 and "UNCHECKED" in out[0]


def test_body_pages_raises_when_no_bibliography_heading(tmp_path, monkeypatch):
    """Without the heading the body's end is unknown; returning a number would invent one."""
    import subprocess
    pdf = _write(tmp_path / "main.pdf")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/pdftotext")

    def fake(cmd, **kw):
        out = "Pages:  3" if cmd[0] == "pdfinfo" else "body text only"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr("subprocess.run", fake)
    with pytest.raises(RuntimeError, match="no 'REFERENCES' heading"):
        verify.body_pages(pdf)


# --- absence, not just presence: superseded values leaking into the paper ----------------

PAPER_VALS = {"90.4": "reachable share, conversation (now 83.5)"}


def test_superseded_value_in_a_retracting_paragraph_is_allowed():
    """The paper cites retracted figures legitimately -- in the sentence that retracts them."""
    text = "An earlier draft reported 90.4\\% here; we retract it.\n"
    assert verify.unmarked_superseded(
        text, PAPER_VALS, markers=verify.PAPER_MARKERS, scope="paragraph") == []


def test_superseded_value_in_a_bare_paragraph_is_flagged():
    text = "The reachable share rises to 90.4\\% on one trace.\n"
    out = verify.unmarked_superseded(
        text, PAPER_VALS, markers=verify.PAPER_MARKERS, scope="paragraph")
    assert len(out) == 1


def test_paragraph_scope_accepts_a_marker_after_the_value():
    """Within one paragraph order carries no meaning: the paper's retractions read
    'described that same band as X --- the repairs disagree by Y'."""
    text = "It described the band as 90.4\\% ---\nthe summary of it was false.\n"
    assert verify.unmarked_superseded(
        text, PAPER_VALS, markers=verify.PAPER_MARKERS, scope="paragraph") == []


def test_section_scope_still_rejects_a_marker_after_the_value():
    """Across sections order DOES carry meaning; this is the AF defect and must stay caught."""
    text = "## H\n\nAttains 90.4\\%.\n\n> **SUPERSEDED.**\n"
    assert len(verify.unmarked_superseded(text, PAPER_VALS, scope="section")) == 1


def test_retraction_in_a_different_paragraph_does_not_license_a_bare_quote():
    """A retraction two paragraphs away is not a label on this number."""
    text = ("The reachable share rises to 90.4\\% on one trace.\n"
            "\n"
            "An earlier draft got this wrong; we retract it.\n")
    out = verify.unmarked_superseded(
        text, PAPER_VALS, markers=verify.PAPER_MARKERS, scope="paragraph")
    assert len(out) == 1


def test_unknown_scope_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError, match="scope must be"):
        verify.unmarked_superseded("x", PAPER_VALS, scope="whole-file")


# --- provenance: mtime is a proxy for "changed", the digest is the thing --------------------

def test_moved_timestamp_with_unchanged_content_is_silent(tmp_path):
    """`touch`, `git checkout` and restore-from-backup all move an mtime without changing a
    byte. A guard that cries wolf on those gets waved past, which is how a real staleness
    slips through."""
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(exp / "a.json", age=OLD)
    m = _write(src / "mod.py", "SOURCE\n", age=NOW)
    manifest = {"a.json": {"mod.py": verify.module_digest(m)}}
    assert verify.stale_artifacts({"a.json": ("mod.py",)}, exp, src, manifest=manifest) == []


def test_changed_content_is_reported_with_both_digests(tmp_path):
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(exp / "a.json", age=OLD)
    m = _write(src / "mod.py", "ORIGINAL\n", age=OLD)
    manifest = {"a.json": {"mod.py": verify.module_digest(m)}}
    _write(src / "mod.py", "EDITED\n", age=NOW)
    out = verify.stale_artifacts({"a.json": ("mod.py",)}, exp, src, manifest=manifest)
    assert len(out) == 1 and "was generated from" in out[0]


def test_absent_provenance_still_reports_and_says_it_rests_on_mtime(tmp_path):
    """A provenance check that silently degrades is worse than one that reports what it sees."""
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(exp / "a.json", age=OLD)
    _write(src / "mod.py", age=NOW)
    out = verify.stale_artifacts({"a.json": ("mod.py",)}, exp, src, manifest={})
    assert len(out) == 1 and "mtime alone" in out[0]


def test_manifest_for_a_different_artifact_does_not_vouch_for_this_one(tmp_path):
    exp, src = tmp_path / "experiments", tmp_path / "src"
    _write(exp / "a.json", age=OLD)
    m = _write(src / "mod.py", age=NOW)
    manifest = {"other.json": {"mod.py": verify.module_digest(m)}}
    out = verify.stale_artifacts({"a.json": ("mod.py",)}, exp, src, manifest=manifest)
    assert len(out) == 1 and "mtime alone" in out[0]


# --- dangling paths: a row that points at nothing ----------------------------------------

def test_missing_artifact_path_is_reported(tmp_path):
    (tmp_path / "experiments").mkdir()
    out = verify.dangling_paths("see experiments/gone.json for this", tmp_path, "DOC.md")
    assert len(out) == 1 and "gone.json" in out[0]


def test_existing_paths_are_silent(tmp_path):
    _write(tmp_path / "experiments" / "here.json")
    assert verify.dangling_paths("see experiments/here.json", tmp_path, "DOC.md") == []


def test_a_path_named_twice_is_reported_once(tmp_path):
    (tmp_path / "experiments").mkdir()
    text = "experiments/gone.json and again experiments/gone.json"
    assert len(verify.dangling_paths(text, tmp_path, "DOC.md")) == 1


def test_superseded_subdirectory_does_not_vouch_for_the_top_level_path(tmp_path):
    """The exact shape of the defect: the artifact was moved to experiments/superseded/ and the
    document kept naming experiments/."""
    _write(tmp_path / "experiments" / "superseded" / "moved.json")
    out = verify.dangling_paths("see experiments/moved.json", tmp_path, "DOC.md")
    assert len(out) == 1


# --- entry points a document promises ----------------------------------------------------

def test_missing_symbol_in_a_real_module_is_reported():
    out = verify.broken_entry_points('python3 -c "from json import no_such_name"', "DOC.md")
    assert len(out) == 1 and "no attribute" in out[0]


def test_unimportable_module_is_reported_not_raised():
    """A guard that raises on a bad promise takes the whole check down with it."""
    out = verify.broken_entry_points('`from stoa.definitely_absent import x`', "DOC.md")
    assert len(out) == 1 and "does not import" in out[0]


def test_working_entry_point_is_silent():
    assert verify.broken_entry_points('`from json import loads`', "DOC.md") == []


def test_each_promise_is_reported_once_even_when_repeated():
    text = "`from json import nope` and again `from json import nope`"
    assert len(verify.broken_entry_points(text, "DOC.md")) == 1


# --- figure coverage derived from what the paper embeds ----------------------------------

def test_embedded_figure_without_a_rule_is_flagged(tmp_path):
    tex = _write(tmp_path / "a.tex", r"\includegraphics[width=1in]{fig_new.pdf}")
    out = verify.figure_coverage_gaps([tex], {}, data_free=())
    assert len(out) == 1 and "no freshness rule covers it" in out[0]


def test_rule_for_an_unembedded_figure_is_flagged(tmp_path):
    """The real defect: four of five rules aimed at figures no .tex includes."""
    tex = _write(tmp_path / "a.tex", "no figures here")
    out = verify.figure_coverage_gaps([tex], {"fig_unused.pdf": ("x.json",)})
    assert len(out) == 1 and "does not embed" in out[0]


def test_data_free_figure_is_exempt_but_must_be_declared(tmp_path):
    tex = _write(tmp_path / "a.tex", r"\includegraphics[width=1in]{fig_schematic.pdf}")
    assert verify.figure_coverage_gaps([tex], {}, data_free=("fig_schematic.pdf",)) == []
    assert len(verify.figure_coverage_gaps([tex], {}, data_free=())) == 1


def test_a_matched_pair_is_silent(tmp_path):
    tex = _write(tmp_path / "a.tex", r"\includegraphics[width=1in]{fig_a.pdf}")
    assert verify.figure_coverage_gaps([tex], {"fig_a.pdf": ("x.json",)}) == []


# --- the EA&B availability URL ------------------------------------------------------------

def test_placeholder_availability_url_is_rejected(tmp_path):
    """ANONYMIZED sat in main.tex through eleven re-verification passes, because every guard
    was pointed at numbers and this is not a number."""
    t = _write(tmp_path / "main.tex",
               r"\renewcommand\vldbavailabilityurl{https://github.com/ANONYMIZED/stoa}")
    out = verify.availability_url_problems(t)
    assert len(out) == 1 and "placeholder" in out[0]


def test_missing_availability_command_is_rejected(tmp_path):
    t = _write(tmp_path / "main.tex", r"\title{A paper with no availability URL}")
    out = verify.availability_url_problems(t)
    assert len(out) == 1 and "EA&B requires" in out[0]


def test_non_https_url_is_rejected(tmp_path):
    """The CfP warns that URLs raising doubt about security of access may jeopardize
    acceptance."""
    t = _write(tmp_path / "main.tex",
               r"\renewcommand\vldbavailabilityurl{http://example.org/stoa}")
    assert len(verify.availability_url_problems(t)) == 1


def test_real_url_passes_without_touching_the_network(tmp_path):
    t = _write(tmp_path / "main.tex",
               r"\renewcommand\vldbavailabilityurl{https://github.com/suanlab/stoa}")
    assert verify.availability_url_problems(t, check_network=False) == []
