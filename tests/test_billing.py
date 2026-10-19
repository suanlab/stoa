"""Which accesses a placement is billed for (notebook AS).

A block first seen at or after the split is CREATED by its first access: its KV is computed then, a
compulsory miss no tier can serve. Every billing path reads `Workload.future_counts`, so it must drop
exactly that access and nothing else. Billing it as a read inflated the oracle's advantage about 3x.
"""
from stoa.environment import ItemState
from stoa.traces import AccessTrace, Workload


def _wl(steps_by_item):
    items = [ItemState(item_id=i, size_bytes=1) for i in steps_by_item]
    traces = {i: AccessTrace(item_id=i, access_steps=sorted(s)) for i, s in steps_by_item.items()}
    accesses = sorted((t, i) for i, s in steps_by_item.items() for t in s)
    return Workload(items=items, accesses=accesses, traces=traces)


def test_a_new_block_reused_twice_bills_two_accesses():
    wl = _wl({"new": [5, 7, 9]})
    assert wl.all_future_accesses(5)["new"] == 3
    assert wl.future_counts(5)["new"] == 2


def test_a_block_born_before_the_split_bills_every_later_access():
    wl = _wl({"old": [1, 6, 8]})
    assert wl.future_counts(5)["old"] == 2 == wl.all_future_accesses(5)["old"]


def test_a_block_created_exactly_at_the_split_is_new():
    wl = _wl({"edge": [5]})
    assert wl.future_counts(5)["edge"] == 0 and wl.all_future_accesses(5)["edge"] == 1


def test_a_block_never_reused_after_creation_bills_nothing():
    wl = _wl({"once": [7], "old": [2]})
    f = wl.future_counts(5)
    assert f["once"] == 0 and f["old"] == 0
