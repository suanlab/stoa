"""The kv-cache-tester universe is a pinned manifest, and loading fails closed (notebook BJ).

Before: the sampled universe was whatever `trace_*.json` sat in the directory (a seed drew from a
different population with 739 files than with 200), and a truncated file was skipped silently.
"""
import hashlib
import json

import pytest

import stoa.kvct as kvct


def _setup(tmp_path, monkeypatch, n=3):
    d = tmp_path / "kvct"
    d.mkdir()
    lines = []
    for i in range(1, n + 1):
        f = d / f"trace_{i:04d}.json"
        f.write_text(json.dumps({"id": f"c{i}", "requests": [{"t": float(i), "hash_ids": [1, 2]}]}))
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    m = tmp_path / "SUMS"
    m.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(kvct, "MANIFEST", m)
    return d


def test_manifest_files_load_and_extra_files_are_ignored(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    (d / "trace_0999.json").write_text(json.dumps({"id": "extra", "requests": []}))
    convs = list(kvct._iter_conversations(str(d)))
    assert [c["_file"] for c in convs] == ["trace_0001.json", "trace_0002.json", "trace_0003.json"]


def test_altered_file_raises(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    (d / "trace_0002.json").write_text("{\"id\": \"c2\", \"requests\": [")      # truncated
    with pytest.raises(ValueError, match="manifest SHA-256"):
        list(kvct._iter_conversations(str(d)))


def test_missing_file_raises(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    (d / "trace_0003.json").unlink()
    with pytest.raises(FileNotFoundError, match="absent"):
        list(kvct._iter_conversations(str(d)))


def test_draws_record_their_files(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    _wl, meta = kvct.load_kvct_with_meta(str(d), limit=2, seed=1)
    assert len(meta["files"]) == 2 and set(meta["files"]) <= {"trace_0001.json", "trace_0002.json", "trace_0003.json"}


def test_shipped_manifest_pins_two_hundred_files():
    m = kvct.manifest()
    assert len(m) == 200 and list(m)[0] == "trace_0001.json" and list(m)[-1] == "trace_0200.json"
