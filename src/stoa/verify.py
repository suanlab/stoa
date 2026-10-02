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
    "leaked_annotations",
    "author_block_problems",
    "bibtex_problems",
    "producer_problems",
    "unbacked_emphasised_numbers",
    "availability_url_problems",
    "figure_coverage_gaps",
    "broken_entry_points",
    "dangling_paths",
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


def body_pages(pdf: Path, end_headings: Sequence[str] = ("REFERENCES",)) -> int:
    """Pages the paper body occupies: the page on which the first end-of-body heading appears.

    The constrained quantity is the body, not the file. PVLDB's limit is "12 pages EXCLUDING
    references"; ICDE's is "12 pages, excluding references and the AI-generated content
    acknowledgement", and the acknowledgement precedes the references, so for ICDE the body ends
    at whichever of the two comes first -- pass both.

    Headings are matched as whole lines after collapsing letter-spacing: IEEEtran sets section
    headings in small caps, which pdftotext renders as "R EFERENCES". The first IEEE build
    therefore found no heading at all -- and raised, as designed, rather than reporting a page
    count it had not measured.
    """
    import re
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
    wanted = {h.upper().replace(" ", "") for h in end_headings}
    for page in range(1, n + 1):
        out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            squashed = re.sub(r"\s+", "", line).upper()
            if squashed in wanted:
                return page
    raise RuntimeError(f"none of {list(end_headings)} found as a heading in {pdf.name}; cannot "
                       "locate the body's end")


def page_limit_violation(pdf: Path, limit: int = 12,
                         end_headings: Sequence[str] = ("REFERENCES",)) -> list[str]:
    """The body over the venue's page limit, or a report that the check could not run."""
    try:
        n = body_pages(pdf, end_headings)
    except (FileNotFoundError, RuntimeError) as exc:
        return [f"PAGE LIMIT UNCHECKED: {exc}"]
    if n > limit:
        return [f"OVER PAGE LIMIT: the body occupies {n} pages against a {limit}-page limit "
                f"(body ends at the first of {list(end_headings)}). Cut before submitting."]
    return []


def dangling_paths(text: str, root: Path, label: str,
                   patterns: Sequence[str] = (r"experiments/[\w.]+\.json",
                                              r"scripts/[\w.]+\.py",
                                              r"docs/[\w.]+\.md")) -> list[str]:
    """Repository paths a document names that do not exist.

    `REPRODUCIBILITY.md` is a claim -> command -> artifact table, and its whole value to a
    committee is that following a row works. It listed `experiments/lrb_sweep.json` for three
    passes after that artifact was moved to `experiments/superseded/` -- so the row pointed at
    nothing, and asserted a claim the paper had already dropped.

    Earlier sweeps for stale prose searched for retracted *vocabulary* and found neither, because
    a dangling path contains no wrong number: it contains no number at all.
    """
    import re

    out = []
    for pat in patterns:
        for m in sorted(set(re.findall(pat, text))):
            if not (root / m).exists():
                out.append(
                    f"{label} names {m}, which does not exist. A reproduction package whose "
                    "rows point at missing files fails at the first step a committee takes.")
    return out


def broken_entry_points(text: str, label: str) -> list[str]:
    """`python3 -c "from X import Y"` commands a document promises that no longer import.

    The reproduction package offers four of these for data acquisition. A rename makes the
    document quietly false: nothing imports them, so nothing fails, and the first thing a
    committee member runs is the thing that breaks. Same class as a dangling path -- the
    document names something that does not exist -- but a grep for missing *files* will not
    find it, because what is missing is a symbol.
    """
    import importlib
    import re

    out = []
    seen = set()
    for mod, names in re.findall(r'from ([\w.]+) import ([\w, ]+)', text):
        for name in (n.strip() for n in names.split(",")):
            if not name or (mod, name) in seen:
                continue
            seen.add((mod, name))
            try:
                m = importlib.import_module(mod)
            except Exception as exc:                     # noqa: BLE001 - report, never raise
                out.append(f"{label} promises `from {mod} import {name}` but {mod} does not "
                           f"import: {type(exc).__name__}: {exc}")
                continue
            if not hasattr(m, name):
                out.append(f"{label} promises `from {mod} import {name}` but {mod} has no "
                           f"attribute {name!r}. A reader runs this before anything else.")
    return out


def figure_coverage_gaps(tex_sources: Iterable[Path], fig_sources: Mapping[str, Sequence[str]],
                         data_free: Sequence[str] = ()) -> list[str]:
    """Figures the paper embeds that no freshness rule covers, and rules aimed at nothing.

    `stale_figures` can only protect figures it is told about, and for two passes the list it
    was given did not match the list the paper uses: it guarded `fig_capacity_ladder.pdf`,
    which appears in no `.tex` file at all, and said nothing about `fig_architecture.pdf`,
    which the paper embeds. `paper/README.md` meanwhile asserted the wrong pair as "the two
    that are used" -- an assertion edited twice without its premise being checked.

    `data_free` names figures drawn from no artifact (a schematic), which are exempt from
    *staleness* but still have to be declared, so that the exemption is a decision on the
    record rather than an omission.
    """
    import re

    used = set()
    for p in tex_sources:
        if p.exists():
            used |= set(re.findall(r"includegraphics\[[^\]]*\]\{([\w.]+\.pdf)\}",
                                   p.read_text()))
    out = []
    for fig in sorted(used - set(fig_sources) - set(data_free)):
        out.append(f"the paper embeds {fig} but no freshness rule covers it. Add it to "
                   "FIG_SOURCES, or declare it data-free if it plots nothing.")
    for fig in sorted(set(fig_sources) - used):
        out.append(f"FIG_SOURCES guards {fig}, which the paper does not embed. A rule aimed "
                   "at an unused figure reads as coverage and is not.")
    return out


def availability_url_problems(main_tex: Path, check_network: bool = False) -> list[str]:
    r"""The EA&B availability URL: still a placeholder, or not anonymously reachable.

    PVLDB is single-blind and requires the artifact in "a publicly accessible archival
    repository". The guidelines add that "URLs that raise doubt about security and anonymity of
    access will cause delays in paper evaluation and might jeopardize acceptance" -- which is
    what a private repository's URL does. This is the one requirement whose failure the CfP
    itself says can cost the paper.

    The placeholder check always runs: `ANONYMIZED` sat in `ldbavailabilityurl` through eleven
    re-verification passes, none of which looked at it, because every guard was pointed at
    numbers. `check_network=True` additionally fetches the URL with no credentials, which is the
    only way to see what a committee sees -- checking it while authenticated tests your own
    access, not theirs.
    """
    import re

    if not main_tex.exists():
        return [f"{main_tex} is missing; the availability URL cannot be checked."]
    text = main_tex.read_text()
    # PVLDB carries the link in \vldbavailabilityurl; the IEEE/ICDE version defines \artifacturl
    # and must also *use* it, or the macro exists while the PDF shows no link at all.
    m = (re.search(r"\\renewcommand\\vldbavailabilityurl\{([^}]*)\}", text)
         or re.search(r"\\newcommand\\artifacturl\{([^}]*)\}", text))
    if not m:
        return ["no artifact URL macro in main.tex. The venue requires the reproducibility "
                "package to be linked in the submission."]
    if "artifacturl" in m.group(0):
        sources = text + "".join(p.read_text() for p in
                                 sorted((main_tex.parent / "sections").glob("*.tex")))
        if r"\url{\artifacturl}" not in sources:
            return [r"\artifacturl is defined but never rendered with \url{\artifacturl}; the "
                    "PDF a reviewer reads does not contain the link."]
    url = m.group(1).strip()
    for placeholder in ("ANONYMIZED", "XXX", "TODO", "example.com"):
        if placeholder in url:
            return [f"\\vldbavailabilityurl is still a placeholder ({url}). EA&B requires a "
                    "live public link at submission."]
    if not url.startswith("https://"):
        return [f"\\vldbavailabilityurl is {url!r}, not https. The CfP warns that URLs raising "
                "doubt about security of access may jeopardize acceptance."]
    if not check_network:
        return []
    import subprocess

    try:
        code = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "-L", "--max-time", "25", url],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}).stdout.strip()
    except Exception as exc:                                  # noqa: BLE001
        return [f"could not fetch {url}: {type(exc).__name__}: {exc}. Verify it by hand before "
                "submitting; do not treat an unreachable check as a pass."]
    if code != "200":
        return [f"{url} returns HTTP {code} without credentials. A committee member sees this, "
                "not what you see while logged in."]
    return []


def unbacked_emphasised_numbers(tex_sources: Iterable[Path], checked_text: str,
                                allow: Sequence[str] = ()) -> list[str]:
    r"""Numbers the paper emphasises that no check ever compared against an artifact.

    The defect this exists for is the largest in the catalogue. `\textbf{96.9\%}` and
    `\textbf{97.6\%}` -- the "97% of the oracle's advantage is unreachable" that the abstract,
    C2, §5.2 and §5.8 all turn on -- appeared in no artifact and in no assertion. Twelve
    re-verification passes had built guards around numbers that *were* in artifacts. Nothing
    had ever asked the reverse question: which emphasised numbers is nothing watching?

    `checked_text` is the concatenation of every value the checker asserts. A `\textbf{...}`
    containing a number whose digits appear nowhere in that text is unbacked. `allow` exempts
    figures that are definitional rather than measured (a page limit, a count of tiers).

    This is a coverage test, not a correctness test: it cannot tell whether a checked number is
    right, only whether anything is looking. `MIN_CHECKS` guarantees the checker does a certain
    *amount* of work; this guarantees it does work on the right things.
    """
    import re

    out = []
    seen = set()
    for p in tex_sources:
        if not p.exists():
            continue
        for m in re.findall(r"\\textbf\{([^}]*\d[^}]*)\}", p.read_text()):
            body = m.strip()
            nums = re.findall(r"\d+(?:[.,]\d+)*", body)
            if not nums:
                continue
            key = (p.name, body)
            if key in seen or body in allow:
                continue
            seen.add(key)
            # Backed if every numeric token in the phrase turns up in the checked values.
            if all(n.replace("{,}", "").replace(",", "") in checked_text.replace(",", "")
                   for n in nums):
                continue
            out.append(
                f"{p.name} emphasises {body!r}, whose digits appear in no checked claim. "
                "Either assert it against an artifact or stop emphasising it; the paper's "
                "headline figure sat unbacked for twelve passes.")
    return out


def producer_problems(depends_on: Mapping[str, Sequence[str]], scripts: Path, src: Path,
                      readers: Sequence[str] = (),
                      producers: Mapping[str, str] | None = None) -> list[str]:
    """Artifacts no script regenerates, and producer imports `DEPENDS_ON` fails to declare.

    `DEPENDS_ON` was maintained by hand and drifted twice over. Four artifacts that load the
    Mooncake traces did not declare `mooncake.py`, so correcting the loader's block size would
    not have marked them stale. And three artifacts -- including the one behind §5.5's LRB band
    -- had no producing script at all: they were assembled in-session, which ICDE's
    Experiment, Analysis and Benchmark category forbids ("MUST provide all artifacts necessary
    to reproduce the results. No exceptions").

    A producer is a script under `scripts` whose source names the artifact, other than the
    `readers` that only consume artifacts; `producers` maps names the producing script builds
    programmatically (e.g. a `_full` / `_lrb` suffix) and so never spells out. Each producer's
    direct `from stoa... import` lines are the modules the artifact encodes, and every one that
    exists under `src` must be declared. Transitive imports are not followed: this catches the
    omissions that actually happened, not every possible one.
    """
    import re

    producers = dict(producers or {})
    out = []
    script_text = {p.name: p.read_text() for p in sorted(scripts.glob("*.py"))
                   if p.name not in readers}
    for art, declared in depends_on.items():
        # The full filename, not its stem: "locomo_power" is a substring of
        # "eval_locomo_powered", and the first version of this audit credited a second producer
        # on that basis.
        names = [n for n, t in script_text.items() if art in t]
        if art in producers:
            names = sorted(set(names) | {producers[art]})
        names = [n for n in names if n in script_text]
        if not names:
            out.append(f"NO PRODUCER: {art} is cited but no script under scripts/ regenerates "
                       "it. An artifact assembled by hand cannot be reproduced by anyone else.")
            continue
        imported = set()
        for n in names:
            for mod in re.findall(r"^\s*from\s+stoa\.([\w.]+)\s+import", script_text[n], re.M):
                imported.add(mod.replace(".", "/") + ".py")
        missing = sorted(m for m in imported if (src / m).exists() and m not in declared)
        if missing:
            out.append(f"UNDECLARED DEPENDENCY: {art} is produced by {', '.join(names)}, which "
                       f"imports {', '.join(missing)}; DEPENDS_ON does not list "
                       f"{'it' if len(missing) == 1 else 'them'}, so a change there would not "
                       "mark this artifact stale.")
    return out


def bibtex_problems(blg: Path) -> list[str]:
    """Errors BibTeX reported in its own log (`main.blg`).

    Three entries in `references.bib` carried `%` comments inside the entry, which BibTeX does
    not support; it reported "I was expecting a `,' or a `}'" on every build, for the life of
    the paper. Nobody saw it, because every build step -- in the Makefile and in every
    re-verification pass -- ran `bibtex main >/dev/null`. The check that the PDF had no
    undefined citations passed throughout, since the entries still resolved. An error stream
    that is discarded is not a stream that is clean.
    """
    import re

    if not blg.exists():
        return [f"{blg.name} is missing; build the paper so BibTeX's own log can be checked."]
    text = blg.read_text(errors="ignore")
    m = re.search(r"\(There (?:was|were) (\d+) error messages?\)", text)
    if m and int(m.group(1)) > 0:
        first = re.search(r"^(I was expecting.*|.*---line \d+ of file .*)$", text, re.M)
        return [f"BibTeX reported {m.group(1)} error(s) in {blg.name}"
                + (f", first: {first.group(1).strip()}" if first else "")
                + ". Its output is easy to discard; this checks the log it writes anyway."]
    return []


def author_block_problems(main_tex: Path,
                          placeholders: Sequence[str] = ("STOA Project", "AUTHOR NAME REQUIRED",
                                                         "AUTHOR STATEMENT REQUIRED")) -> list[str]:
    r"""Placeholder authorship in a single-blind submission.

    `\author{STOA Project}` -- the repository's git user name, not a person -- sat on page 1
    through every PVLDB re-verification pass. PVLDB's guidelines say "authors MUST include their
    names and affiliations on the first page"; ICDE is likewise single-blind. Nothing checked,
    because every guard was pointed at numbers. The ICDE acknowledgement also carries a marked
    slot for the author's own statement of their role, which only the author can write.
    """
    if not main_tex.exists():
        return [f"{main_tex} is missing."]
    import re

    # Strip LaTeX comments first: the comment explaining why this guard exists names the old
    # placeholder, and the first version of the guard flagged its own documentation.
    text = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in main_tex.read_text().splitlines())
    return [f"main.tex still contains the placeholder {p!r}. A single-blind submission needs real "
            "author names, and the AI-use acknowledgement needs the author's own statement."
            for p in placeholders if p in text]


def leaked_annotations(pdf: Path,
                       markers: Sequence[str] = ("Verified:", "TODO", "FIXME", "XXX",
                                                 "REQUIRED")) -> list[str]:
    """Internal notes that reached the rendered PDF.

    `references.bib` carried verification notes in its `note` field ("Verified: USENIX FAST'03.
    Adaptively balances recency against frequency; ..."). ACM's style and IEEEtran both render
    `note`, so the paper's reference list published them -- in the PVLDB version as well, where
    nothing noticed for twelve passes, because every check read the LaTeX source and none read
    the PDF a reviewer reads. Other entries carry Korean-language reminders ("arXiv ID 재확인
    권장"); they happened not to be cited, so they happened not to render.

    Checked on the output rather than inferred from the source: any marker above, and any Hangul
    syllable in an English-language paper, is a leaked annotation.
    """
    import re
    import shutil
    import subprocess

    if not pdf.exists():
        return [f"{pdf} is missing; build it before checking what it publishes."]
    if shutil.which("pdftotext") is None:
        return ["pdftotext is unavailable, so the rendered PDF cannot be checked for leaked "
                "annotations; do not treat this as a pass."]
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True,
                          check=True).stdout
    out = []
    for mk in markers:
        if mk in text:
            ctx = text[max(0, text.index(mk) - 40): text.index(mk) + 60].replace("\n", " ")
            out.append(f"the rendered PDF contains {mk!r}: ...{ctx}...")
    hangul = re.findall(r"[가-힣]+", text)
    if hangul:
        out.append(f"the rendered PDF contains Hangul ({', '.join(sorted(set(hangul))[:5])}); in an "
                   "English-language paper that is a leaked internal note.")
    return out
