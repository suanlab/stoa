"""The Bailian -> Mooncake-schema conversion behind the H7 replication (scripts/run_bailian.py).

A Bailian record is a superset of a Mooncake record, so conversion is a projection. What must not
change is what the producers measure: the request sequence, every block id in order, and the
timestamps. And the result must load through the repository's own loader at 16 tokens per block,
whose identity check is what proves the block size, and be refused at Mooncake's 512.
"""
import json
import math
import sys
from pathlib import Path

import pytest

from stoa import mooncake

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_bailian  # noqa: E402  (module import has no side effects; patches live in worker())


def _row(chat, t, n_in, hashes, parent=-1, turn=1, typ="text"):
    assert len(hashes) == math.ceil(n_in / 16)
    return {"chat_id": chat, "parent_chat_id": parent, "timestamp": t, "input_length": n_in,
            "output_length": 7, "type": typ, "turn": turn, "hash_ids": hashes}


ROWS = [_row(0, 0.0, 40, [0, 1, 2]),
        _row(1, 0.206, 17, [3, 4], typ="image"),
        _row(2, 1.5, 64, [0, 1, 2, 5], parent=0, turn=2),        # reuses chat 0's prefix
        _row(3, 1.5, 16, [6]),                                    # equal timestamps are allowed
        _row(4, 3700.25, 33, [0, 1, 7], parent=2, turn=3)]       # past the one-hour mark


def _write(tmp_path, rows, name="bailian.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def test_conversion_preserves_requests_blocks_and_timestamps(tmp_path):
    src, dst = _write(tmp_path, ROWS), tmp_path / "out.jsonl"
    info = run_bailian.convert(src, dst)
    out = _read(dst)
    assert info["requests"] == len(out) == len(ROWS)
    assert [r["hash_ids"] for r in out] == [r["hash_ids"] for r in ROWS]
    assert [r["timestamp"] for r in out] == [r["timestamp"] for r in ROWS]
    assert [r["input_length"] for r in out] == [r["input_length"] for r in ROWS]
    assert all(set(r) == set(run_bailian.MOONCAKE_FIELDS) for r in out)


def test_converted_trace_loads_at_16_tokens_with_the_same_accesses(tmp_path, monkeypatch):
    dst = tmp_path / "out.jsonl"
    run_bailian.convert(_write(tmp_path, ROWS), dst)
    monkeypatch.setattr(mooncake, "BLOCK_TOKENS", 16)
    wl, extra = mooncake.load_mooncake(str(dst), with_wallclock=True)
    assert len(wl.accesses) == sum(len(r["hash_ids"]) for r in ROWS)
    assert {i.item_id for i in wl.items} == {f"b{h}" for r in ROWS for h in r["hash_ids"]}
    assert wl.traces["b0"].access_steps == [0, 2, 4]             # request index = timestep
    assert [extra["wall"][t] for t in range(len(ROWS))] == [r["timestamp"] for r in ROWS]
    assert all(i.size_bytes == 16 * mooncake.BYTES_PER_TOKEN for i in wl.items)


def test_converted_trace_is_refused_at_mooncakes_block_size(tmp_path):
    """The loader's identity check, not a citation, is what fixes the block size."""
    dst = tmp_path / "out.jsonl"
    run_bailian.convert(_write(tmp_path, ROWS), dst)
    assert mooncake.BLOCK_TOKENS == 512
    with pytest.raises(ValueError, match="does not match the trace"):
        mooncake.load_mooncake(str(dst))


def test_time_prefix_keeps_a_prefix_not_a_sample(tmp_path):
    dst = tmp_path / "out.jsonl"
    info = run_bailian.convert(_write(tmp_path, ROWS), dst, hours=1.0)
    out = _read(dst)
    assert info["requests"] == 4 and [r["hash_ids"] for r in out] == [r["hash_ids"] for r in ROWS[:4]]


def test_backwards_timestamp_is_refused(tmp_path):
    rows = [ROWS[0], ROWS[2], ROWS[1]]
    with pytest.raises(ValueError, match="timestamp"):
        run_bailian.convert(_write(tmp_path, rows), tmp_path / "out.jsonl")


def test_block_count_violating_16_tokens_is_refused(tmp_path):
    bad = dict(ROWS[0], input_length=1000)                       # 3 blocks cannot hold 1000 tokens
    with pytest.raises(ValueError, match="tokens/block"):
        run_bailian.convert(_write(tmp_path, [bad]), tmp_path / "out.jsonl")


def test_structure_counts_singleton_blocks(tmp_path):
    dst = tmp_path / "out.jsonl"
    run_bailian.convert(_write(tmp_path, ROWS), dst)
    s = run_bailian.structure(dst)
    # blocks 0,1 seen 3x, 2 seen 2x; 3,4,5,6,7 once -> 5 of 8 blocks are singletons
    assert (s["requests"], s["blocks"], s["accesses"]) == (5, 8, 13)
    assert s["singleton_block_frac"] == round(5 / 8, 4)


def test_regrouping_is_a_prefix_chain_that_satisfies_the_coarser_identity(tmp_path, monkeypatch):
    """`group` is the block-size sensitivity variant: 2 x 16-token ids -> one 32-token block whose
    id depends on everything before it, as Mooncake's do. Only the default keeps upstream ids."""
    rows = [_row(0, 0.0, 64, [0, 1, 2, 3]),
            _row(1, 1.0, 64, [0, 1, 2, 9]),          # same first 32 tokens, different second half
            _row(2, 2.0, 64, [5, 6, 2, 3])]          # same second half, different prefix
    dst = tmp_path / "g.jsonl"
    run_bailian.convert(_write(tmp_path, rows), dst, group=2)
    ids = [r["hash_ids"] for r in _read(dst)]
    assert ids[0][0] == ids[1][0]                    # a shared 32-token prefix is one block
    assert ids[0][1] != ids[1][1]
    assert ids[2][1] != ids[0][1]                    # equal content after a different prefix is not
    monkeypatch.setattr(mooncake, "BLOCK_TOKENS", 32)
    assert len(mooncake.load_mooncake(str(dst)).accesses) == 6


def test_library_bills_only_servable_accesses_on_bailian_schema(tmp_path, monkeypatch):
    """Notebook AS: future_counts drops a post-split block's creation access; the run_bailian
    patch is then a no-op (it checks the library's marker), so nothing is subtracted twice."""
    from stoa.traces import Workload
    monkeypatch.setattr(Workload, "future_counts", Workload.future_counts)   # restored after
    dst = tmp_path / "out.jsonl"
    run_bailian.convert(_write(tmp_path, ROWS), dst)
    monkeypatch.setattr(mooncake, "BLOCK_TOKENS", 16)
    wl = mooncake.load_mooncake(str(dst))
    raw = wl.all_future_accesses(2)
    before = wl.future_counts(2)
    run_bailian.patch_birth_excluded()
    assert wl.future_counts(2) == before                  # the patch must not subtract again
    assert raw["b0"] == before["b0"] == 2                 # born before the split: unchanged
    assert raw["b5"] == 1 and before["b5"] == 0           # born at the split, never reused
    assert raw["b7"] == 1 and before["b7"] == 0           # born after the split
    stat = run_bailian.post_split_never_reused(dst, [0.5])["0.5"]
    assert stat == {"new_blocks": 3, "never_reused_frac": 1.0}   # b5, b6, b7 each seen once


def test_both_billing_modes_are_explicit_whatever_the_library_default(tmp_path, monkeypatch):
    """run_bailian's workers install the birth-billed counts first and the §AS correction on top
    when asked, so `--birthx` and its absence each mean one thing even after the library default
    changed mid-replication (2026-10-19, src/stoa/traces.py)."""
    from stoa.traces import Workload
    monkeypatch.setattr(Workload, "future_counts", Workload.future_counts)   # restored after
    dst = tmp_path / "out.jsonl"
    run_bailian.convert(_write(tmp_path, ROWS), dst)
    monkeypatch.setattr(mooncake, "BLOCK_TOKENS", 16)
    wl = mooncake.load_mooncake(str(dst))
    run_bailian.patch_birth_billed()
    billed = wl.future_counts(2)
    assert billed == wl.all_future_accesses(2)
    assert (billed["b0"], billed["b5"], billed["b7"]) == (2, 1, 1)
    run_bailian.patch_birth_excluded()
    excluded = wl.future_counts(2)
    assert (excluded["b0"], excluded["b5"], excluded["b7"]) == (2, 0, 0)
    run_bailian.patch_birth_excluded()                    # idempotent: never subtracts twice
    assert wl.future_counts(2) == excluded
