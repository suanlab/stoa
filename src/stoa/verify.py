"""Freshness and consistency guards for the reproducibility artifact.

Each function here exists because a specific defect reached a draft of the paper and no check
caught it. They are pure -- taking paths and mappings rather than reading module globals -- so
that `tests/test_verify.py` can exercise every one on a synthetic tree. That matters more than
it looks: for four re-verification passes these guards lived inline in
`scripts/check_paper_numbers.py`, where a refactor could have deleted any of them and every
pass would still have reported clean. A guard that can vanish silently is the same failure it
was written to prevent.

The chain a result travels, and the guard on each edge:

    src/*.py  --stale_artifacts-->  experiments/*.json  --stale_figures-->  paper/figs/*.pdf
                                                                                    |
                                            stale_paper_pdf                         v
                                                    ...  paper/sections/*.tex  -->  main.pdf

and, off to the side, the surfaces that restate a result rather than deriving it:
`prose_drift` (assertions, which are updated) and `unmarked_superseded` (records, which are
banner-marked and left intact).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "body_pages",
    "page_limit_violation",
    "stale_artifacts",
    "undeclared_artifacts",
    "stale_figures",
    "stale_paper_pdf",
    "prose_drift",
    "unmarked_superseded",
    "SUPERSESSION_MARKERS",
]

# "paper carried | regenerated" is an explicit before/after column header -- the value is
# labelled superseded, just in different words. Accepting it is not a weakening.
SUPERSESSION_MARKERS = ("SUPERSEDED", "RETRACTED", "FALSIFIED", "do not cite", "paper carried")


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def stale_artifacts(depends_on: Mapping[str, Sequence[str]], exp: Path, src: Path) -> list[str]:
    """Artifacts older than a module whose behaviour they encode.

    The defect: three artifacts the paper cited predated a tie-break fix to the routines that
    produced them, two of them denominators. The checker compared artifacts to the *paper* and
    never to the *source*, so fixing the code did not fix -- or flag -- the stored results.
    """
    out = []
    for art, mods in depends_on.items():
        ap = exp / art
        if not ap.exists():
            continue
        for mod in mods:
            mp = src / mod
            if mp.exists() and _mtime(mp) > _mtime(ap):
                out.append(
                    f"STALE ARTIFACT: {art} predates {mp}. Regenerate it before trusting any "
                    "number it feeds; the code was fixed and the result was not.")
    return out


def undeclared_artifacts(loaded: Iterable[str], depends_on: Mapping[str, Sequence[str]],
                         exp: Path) -> list[str]:
    """Artifacts the checker reads but never declares dependencies for.

    Without this, adding an artifact and forgetting to declare it exempts it from staleness
    checking permanently, and nothing says so. It found `sampling_study.json` already in that
    state on the day it was written.
    """
    missing = sorted({n for n in loaded if n not in depends_on and (exp / n).exists()})
    if not missing:
        return []
    return ["artifacts read but absent from DEPENDS_ON, so never staleness-checked: "
            + ", ".join(missing) + ". Declare the modules each depends on."]


def stale_figures(fig_sources: Mapping[str, Sequence[str]], figs: Path, exp: Path) -> list[str]:
    """Figures older than the artifacts they plot.

    The paper embeds the PDF, not the JSON, so a number-checker that greps the LaTeX source
    cannot see a plot at all. Found the paper's two figures fifteen days behind their data:
    corrected text beside an uncorrected picture of the retracted result.
    """
    out = []
    for fig, arts in fig_sources.items():
        fp = figs / fig
        if not fp.exists():
            continue
        for art in arts:
            ap = exp / art
            if ap.exists() and _mtime(ap) > _mtime(fp):
                out.append(
                    f"STALE FIGURE: {fp.name} predates {art}. Run scripts/make_figures.py; the "
                    "paper embeds the PDF, not the JSON, so the text can be corrected while the "
                    "plot still shows the retracted number.")
                break
    return out


def stale_paper_pdf(pdf: Path, sources: Iterable[Path]) -> list[str]:
    """The built PDF older than any source it is built from.

    The last edge of the chain. A correction landing in the .tex is not a correction until the
    PDF is rebuilt, and the PDF is what gets attached to a submission.
    """
    if not pdf.exists():
        return []
    behind = sorted(str(s) for s in sources if s.exists() and _mtime(s) > _mtime(pdf))
    if not behind:
        return []
    return [f"STALE PDF: {pdf.name} predates {', '.join(behind)}. Rebuild before submitting; "
            "the PDF is the artifact a reviewer reads, not the source."]


def prose_drift(files: Iterable[str], headline: Mapping[str, str], root: Path,
                trigger: str) -> list[str]:
    """Released prose restating a result without its current numbers.

    These files are *assertions*, so the repair is an update. Twice a correction landed in the
    .tex and not in the markdown, most recently leaving the repository's front page advertising
    two numbers a re-verification pass had just falsified.

    `trigger` keeps the check honest: only files that actually restate the result are held to
    it, so an unrelated README is not forced to quote numbers it never mentions.
    """
    out = []
    for fname in files:
        fp = root / fname
        if not fp.exists():
            continue
        text = fp.read_text()
        if trigger not in text:
            continue
        for label, val in headline.items():
            if val not in text:
                out.append(
                    f"{fname} restates the result but does not contain {val} ({label}). "
                    "Released prose drifts from the paper silently; it is the first thing a "
                    "reader sees and nothing else checks it.")
    return out


def unmarked_superseded(text: str, values: Mapping[str, str], label: str = "notebook",
                        markers: Sequence[str] = SUPERSESSION_MARKERS) -> list[str]:
    """Superseded values stated in a record without a supersession marker.

    A lab notebook must keep superseded measurements; editing them out destroys the record,
    which is the reason the file exists. So this is a *record*, and the repair is a banner
    rather than a rewrite. A value counts as marked when a marker appears anywhere in its
    section -- from the nearest preceding heading through the line itself.

    The defect: the notebook stated a falsified conclusion in the present tense a hundred and
    fifty lines above its own retraction of it.
    """
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for val, what in values.items():
            if val not in line:
                continue
            start = 0
            for j in range(i, -1, -1):
                if lines[j].startswith("#"):
                    start = j
                    break
            # Case-insensitive: prose says "the claim AC falsified", headers say "SUPERSEDED".
            section = "\n".join(lines[start:i] + [line]).lower()
            if not any(m.lower() in section for m in markers):
                out.append(
                    f"{label}:{i + 1} states the superseded value {val} ({what}) in a section "
                    "carrying no SUPERSEDED/RETRACTED marker. Add a banner rather than editing "
                    "the record.")
    return out


def body_pages(pdf: Path, heading: str = "REFERENCES") -> int:
    """Pages the paper body occupies -- the page the bibliography starts on.

    PVLDB's limit is "12 pages EXCLUDING references", so the constrained quantity is this, not
    the page count of the file. For five re-verification passes the check read `pdfinfo` and
    compared the whole PDF against 12. That agreed with the truth only while the references
    happened to fit on the body's last page; it can fail in both directions, passing a paper
    whose body runs to thirteen and failing a compliant one whose references spill over.

    Raises rather than returning a sentinel: a page-limit check that cannot run must not look
    like a page-limit check that passed.
    """
    import shutil
    import subprocess

    if not pdf.exists():
        raise FileNotFoundError(pdf)
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext is unavailable, so the page limit cannot be measured. Install poppler-utils; "
            "do not treat this as a pass.")
    n = int(subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True,
                           check=True).stdout.split("Pages:")[1].split()[0])
    for page in range(1, n + 1):
        out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"],
                             capture_output=True, text=True, check=True).stdout
        if heading in out:
            return page
    raise RuntimeError(f"no {heading!r} heading found in {pdf.name}; cannot locate the body's end")


def page_limit_violation(pdf: Path, limit: int = 12) -> list[str]:
    """The body over the venue's page limit, or a report that the check could not run."""
    try:
        n = body_pages(pdf)
    except (FileNotFoundError, RuntimeError) as exc:
        return [f"PAGE LIMIT UNCHECKED: {exc}"]
    if n > limit:
        return [f"OVER PAGE LIMIT: the body occupies {n} pages against a {limit}-page limit "
                "excluding references. Cut before submitting."]
    return []
