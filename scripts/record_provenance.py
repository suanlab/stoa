#!/usr/bin/env python3
"""Record which module contents each artifact was generated from.

`experiments/.provenance.json` maps artifact -> {module: sha256}. `check_paper_numbers.py`
consults it when the mtime check fires: a matching digest means the timestamp moved without the
content changing (a `touch`, a `git checkout`, a restore after mutation testing) and the artifact
is fine; a differing digest is real staleness.

**Run this only when the artifacts are genuinely current** -- immediately after regenerating
them, or after confirming with `git diff` that the modules are unchanged from the commit the
artifacts were validated in. Running it to silence a complaint defeats the guard entirely, which
is why it is a separate command and not something the checker does for itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stoa import verify  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    spec = ROOT / "scripts" / "check_paper_numbers.py"
    ns: dict = {}
    src = spec.read_text()
    start = src.index("DEPENDS_ON = {")
    end = src.index("}", start) + 1
    exec(src[start:end], ns)
    depends_on = ns["DEPENDS_ON"]

    # Only the modules some artifact actually depends on. A dirty `verify.py` cannot make an
    # artifact stale, and refusing on it would train the user to pass a --force nobody checks.
    deps = sorted({f"src/stoa/{m}" for mods in depends_on.values() for m in mods})
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *deps],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if dirty:
        print("modules that artifacts depend on have uncommitted changes:\n" + dirty)
        print("\nRefusing: recording provenance from modified sources would stamp artifacts as "
              "current against code they were not generated from. Regenerate the artifacts, or "
              "commit/revert the modules first.")
        return 1

    out = {}
    for art, mods in depends_on.items():
        if not (ROOT / "experiments" / art).exists():
            continue
        out[art] = {m: verify.module_digest(ROOT / "src" / "stoa" / m)
                    for m in mods if (ROOT / "src" / "stoa" / m).exists()}
    dest = ROOT / "experiments" / ".provenance.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} for {len(out)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
