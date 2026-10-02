"""The Mooncake loader's block-size identity (§AO).

`BLOCK_TOKENS` was 256 for the life of the PVLDB paper, cited to the Mooncake paper; the release
README says 512 and so does the data: `len(hash_ids) == ceil(input_length / 512)` for all 35,639
production requests and for none at 256. The loader now enforces the identity per request, so a
wrong constant fails on the first line it reads instead of surviving as a citation.
"""
import json
import math

import pytest

from stoa import mooncake


def _trace(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_block_size_is_the_one_the_release_and_the_data_agree_on():
    assert mooncake.BLOCK_TOKENS == 512


def test_consistent_trace_loads(tmp_path):
    rows = [{"timestamp": 0, "input_length": 1000, "output_length": 5, "hash_ids": [1, 2]},
            {"timestamp": 3, "input_length": 1024, "output_length": 5, "hash_ids": [1, 2]},
            {"timestamp": 6, "input_length": 1025, "output_length": 5, "hash_ids": [1, 2, 3]}]
    wl = mooncake.load_mooncake(_trace(tmp_path, rows))
    assert len(wl.items) == 3 and len(wl.accesses) == 7


def test_trace_with_a_different_block_size_is_refused(tmp_path):
    """A 256-token trace has twice the hash_ids a 512-token loader predicts."""
    rows = [{"timestamp": 0, "input_length": 1000, "output_length": 5,
             "hash_ids": list(range(math.ceil(1000 / 256)))}]
    with pytest.raises(ValueError, match="does not match the trace"):
        mooncake.load_mooncake(_trace(tmp_path, rows))


def test_wrong_constant_is_caught_on_real_shaped_input(tmp_path, monkeypatch):
    """The regression this guards: put 256 back and a 512-token trace must not load."""
    monkeypatch.setattr(mooncake, "BLOCK_TOKENS", 256)
    rows = [{"timestamp": 0, "input_length": 6955, "output_length": 52,
             "hash_ids": list(range(14))}]              # the upstream README's own example
    with pytest.raises(ValueError):
        mooncake.load_mooncake(_trace(tmp_path, rows))
