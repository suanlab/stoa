#!/usr/bin/env python3
"""Record the content digests of paper/main.pdf and every source it is built from.

Run immediately after a successful build (`make paper` does). The manifest is committed, so a
fresh clone -- where `git clone` leaves file times in arbitrary order -- can confirm that the
shipped PDF matches the shipped sources instead of failing on timestamps (verify.stale_paper_pdf).

Refuses when a source is newer than the PDF: that means the build did not run after the last
edit, and recording it would certify a stale PDF.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stoa import verify  # noqa: E402


def main() -> int:
    paper = ROOT / "paper"
    pdf = paper / "main.pdf"
    sources = [paper / "main.tex", *sorted((paper / "sections").glob("*.tex")),
               *sorted((paper / "figs").glob("*.pdf")), ROOT / "docs" / "references.bib",
               *(paper / f for f in ("acmart.cls", "pvldb.sty", "ACM-Reference-Format.bst"))]
    stale = verify.stale_paper_pdf(pdf, sources)
    if stale:
        print("Refusing: " + stale[0])
        return 1
    out = paper / ".build_manifest.json"
    out.write_text(json.dumps(verify.pdf_build_manifest(pdf, sources), indent=2, sort_keys=True)
                   + "\n")
    print(f"wrote {out.relative_to(ROOT)} for {len(sources)} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
