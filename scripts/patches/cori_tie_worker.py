# Runs under the Python 2.7 that Cori's simulator (GTkernel/cori-sim, IPDPS'21) is written for; see
# run_cori_tie_audit.py. Changes one thing: the order among pages with equal binned counts in
# AddressSpace.get_l2_hot_pages, where the simulator ranks a period's touched pages with
# np.argsort(counts)[::-1] (numpy's default, unstable) and moves the top ones to fast memory.
#
#   python2 cori_tie_worker.py CORI_DIR candidates TRACE
#   python2 cori_tie_worker.py CORI_DIR run TRACE POLICY REQS_PER_PERIOD MODE
#
# MODE: default (the simulator as published) | first (lower page id first; ids are assigned in first-touch
# order) | last (higher id first) | randN (a random order, seed N). Prints one JSON line.
from __future__ import print_function
import json
import os
import sys

CORI = sys.argv[1]
sys.path.insert(0, CORI)
os.chdir(CORI)
import numpy as np                                   # noqa: E402
import run_cori                                      # noqa: E402  (its __main__ block does not run on import)
from sim.perf_model import PerfModel, Profile        # noqa: E402
import sim.memory as M                               # noqa: E402

if sys.argv[2] == "candidates":
    bins, heights, prof = run_cori.get_reuse_hist(sys.argv[3], 1000)
    dom = run_cori.get_dominant_reuse(bins, heights)
    print(json.dumps({"trace": os.path.basename(sys.argv[3]), "dominant_reuse": dom, "num_reqs": prof.traffic.num_reqs,
                      "candidates": run_cori.get_candidates(dom, prof.traffic.num_reqs / 2.0)}))
    sys.exit(0)

trace, policy, reqs, mode = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
STATE = {"rng": np.random.RandomState(int(mode[4:])) if mode.startswith("rand") else None,
         "periods": 0, "over_capacity": 0, "cut_in_tie": 0}
_published = M.AddressSpace.get_l2_hot_pages
_argsort = np.argsort


def get_l2_hot_pages(self, curr_ep, policy):
    def ordered(a, *args, **kwargs):
        a = np.asarray(a)
        STATE["periods"] += 1
        if len(a) > self.l1_pages:
            s = np.sort(a)[::-1]
            STATE["over_capacity"] += 1
            STATE["cut_in_tie"] += int(s[self.l1_pages - 1] == s[self.l1_pages])
        if mode == "default":
            return _argsort(a, *args, **kwargs)
        ids = np.arange(len(a))                      # position = order of page id (the page list is in id order)
        if mode == "first":                          # ascending sort, reversed by the caller: lower id first
            return np.lexsort((-ids, a))
        if mode == "last":
            return np.lexsort((ids, a))
        return np.lexsort((STATE["rng"].permutation(len(a)), a))
    np.argsort = ordered
    try:
        return _published(self, curr_ep, policy)
    finally:
        np.argsort = _argsort


M.AddressSpace.get_l2_hot_pages = get_l2_hot_pages
prof = Profile(trace)
prof.init()
sim = PerfModel(prof, "Fast:NearSlow", policy, 0.2, reqs)
sim.init()
sim.run()
print(json.dumps({"trace": os.path.basename(trace), "policy": policy, "reqs_per_period": reqs, "mode": mode,
                  "slowdown_pct": sim.stats["Slowdown_from_all_fast"], "fast_hitrate_pct": sim.stats["Fast_Hitrate"],
                  "periods_ranked": STATE["periods"], "periods_over_capacity": STATE["over_capacity"],
                  "periods_cut_in_tie": STATE["cut_in_tie"]}))
