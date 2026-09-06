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
    "PAPER_MARKERS",
    "module_digest",
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

# The paper cites superseded numbers legitimately, but only where it is retracting them. These
# are the phrases it uses to do that; anything else quoting a retracted figure is a leak.
PAPER_MARKERS = ("retract", "earlier draft", "previously said", "superseded", "falsified",
                 "we withhold", "no longer", "was false", "is neither", "wrong")


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def module_digest(path: Path) -> str:
    """SHA-256 of a source file, as the thing an artifact actually depends on."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def stale_artifacts(depends_on: Mapping[str, Sequence[str]], exp: Path, src: Path,
                    manifest: Mapping[str, Mapping[str, str]] | None = None) -> list[str]:
    """Artifacts older than a module whose behaviour they encode.

    The defect: three artifacts the paper cited predated a tie-break fix to the routines that
    produced them, two of them denominators. The checker compared artifacts to the *paper* and
    never to the *source*, so fixing the code did not fix -- or flag -- the stored results.

    mtime is a *proxy* for "changed", not the thing itself. It fires falsely on `touch`, on a
    `git checkout`, on restoring a file from a backup -- all of which happened here during
    mutation testing -- and it can miss a write that preserves the timestamp. A false positive
    is not benign: this guard's advice costs a 35-minute regeneration, and a guard that cries
    wolf is a guard that gets waved past, which is how a real staleness would slip through.

    So `manifest` records, per artifact, the digest each module had when that artifact was
    generated. When the mtime check fires, a matching digest means the timestamp lied and the
    artifact is fine. A *differing* digest is real staleness and is reported with the digests
    named. Without a manifest the check falls back to mtime alone and says so, because a
    provenance check that silently degrades is worse than one that reports what it can see.
    """
    out = []
    for art, mods in depends_on.items():
        ap = exp / art
        if not ap.exists():
            continue
        recorded = (manifest or {}).get(art, {})
        for mod in mods:
            mp = src / mod
            if not (mp.exists() and _mtime(mp) > _mtime(ap)):
                continue
            if mod in recorded:
                if recorded[mod] == module_digest(mp):
                    continue  # mtime moved, content did not
                out.append(
                    f"STALE ARTIFACT: {art} was generated from {mod} at "
                    f"{recorded[mod][:12]}, which is now {module_digest(mp)[:12]}. Regenerate "
                    "it before trusting any number it feeds.")
            else:
                out.append(
                    f"STALE ARTIFACT: {art} predates {mp} and has no recorded provenance for "
                    f"{mod}, so this rests on mtime alone. Regenerate it, or record its "
                    "provenance with scripts/record_provenance.py if the timestamp is lying.")
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
                        markers: Sequence[str] = SUPERSESSION_MARKERS,
                        scope: str = "section") -> list[str]:
    """Superseded values stated in a record without a supersession marker.

    A lab notebook must keep superseded measurements; editing them out destroys the record,
    which is the reason the file exists. So this is a *record*, and the repair is a banner
    rather than a rewrite. A value counts as marked when a marker appears anywhere in its
    section -- from the nearest preceding heading through the line itself.

    `scope` sets how far the marker may sit from the value. ``"section"`` searches from the
    nearest preceding markdown heading -- right for a notebook, where a banner heads the whole
    superseded section. ``"paragraph"`` searches only the blank-line-delimited block the value
    sits in, which is what the paper needs: it cites superseded numbers legitimately, but only
    in the sentence that retracts them, and a retraction two sections away is not a label.

    The defect: the notebook stated a falsified conclusion in the present tense a hundred and
    fifty lines above its own retraction of it. And the checker, being a *presence* test --
    "does the correct value appear anywhere?" -- passed all 68 numeric checks with the two
    falsified values sitting in the abstract, because the correct ones appeared in section 5.
    """
    if scope not in ("section", "paragraph"):
        raise ValueError(f"scope must be 'section' or 'paragraph', not {scope!r}")
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for val, what in values.items():
            if val not in line:
                continue
            start = 0
            for j in range(i, -1, -1):
                if lines[j].startswith("#") if scope == "section" else not lines[j].strip():
                    start = j
                    break
            # Section scope requires the marker to PRECEDE the value: a banner heads the
            # superseded material, and one further down the file is the §AF defect. Within a
            # single paragraph order carries no such meaning -- the paper's retractions read
            # "described that same band as X --- the repairs disagree by Y" -- so the whole
            # paragraph counts.
            end = i + 1
            if scope == "paragraph":
                for j in range(i + 1, len(lines)):
                    if not lines[j].strip():
                        break
                    end = j + 1
            # Case-insensitive: prose says "the claim AC falsified", headers say "SUPERSEDED".
            section = "\n".join(lines[start:end]).lower()
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
