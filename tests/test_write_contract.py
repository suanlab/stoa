"""The serving-replay collector reads the right requests and the right stores (notebook §BM).

Its numbers come from log lines: a request can be looked up more than once (the last lookup is the one
it was served with), stores carry no request id (only those after the first measured request's lookup
count), and the offloading connector's hit line is cumulative (the last one is the total).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_write_contract as wc  # noqa: E402

PRE = "\x1b[32;20m[{ts}] LMCache INFO:\x1b[0m "


def _lookup(ts, rid, tot, hit, need):
    return PRE.format(ts=ts) + f"Reqid: {rid}, Total tokens {tot}, LMCache hit tokens: {hit}, need to load: {need}"


def _store(ts, n, ms):
    return PRE.format(ts=ts) + f"Stored {n} out of total {n} tokens. size: 0.1 gb, cost {ms} ms, throughput: 7 GB/s"


def _write(tmp_path, name, recs, log):
    stem = tmp_path / name
    Path(f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    Path(f"{stem}.server.log").write_text("\n".join(log) + "\n")
    return stem


def test_lmcache_warmup_boundary_last_lookup_and_stores(tmp_path):
    recs = [{"i": 0, "warmup": True, "ttft_ms": 5.0, "err": None},
            {"i": 1, "warmup": False, "ttft_ms": 7.0, "err": None},
            {"i": 2, "warmup": False, "ttft_ms": None, "err": "ServerDisconnectedError()"}]
    log = [_lookup("2026-11-21 10:00:00,000", "a", 512, 0, -16),
           _store("2026-11-21 10:00:00,500", 512, "4.0"),         # the warm-up request's store: not counted
           _lookup("2026-11-21 10:00:01,000", "b", 1024, 0, 0),
           _lookup("2026-11-21 10:00:01,200", "b", 1024, 512, 480),  # looked up again: this one counts
           _store("2026-11-21 10:00:01,500", 512, "6.0"),
           _lookup("2026-11-21 10:00:02,000", "c", 256, 256, 256)]
    run = wc.parse_run(_write(tmp_path, "lmcache_store-all_x0.2", recs, log), warm=1)
    assert run["ttft_ms"] == [7.0, None] and run["errors"] == 1
    assert run["lmcache_store"] == {"ms_measured": 6.0, "stores": 1, "tokens": 512}
    # b: 32 GPU-local + 480 host; c: 0 GPU-local + 256 host
    assert run["hits"] == {"reqs": 2, "tokens": 1280, "gpu_hit": 32, "host_hit": 736}


def test_offload_reads_the_last_cumulative_line(tmp_path):
    recs = [{"i": 0, "warmup": False, "ttft_ms": 3.0, "err": None}]
    log = ["STOA_SKIPSTORE calls=100 store_jobs=5 stored_gpu_blocks=40 skipped_reqs=1",
           "STOA_HITS reqs=3 measured_reqs=1 measured_tokens=100 measured_gpu_hit=10 measured_host_hit=20",
           "STOA_SKIPSTORE calls=200 store_jobs=9 stored_gpu_blocks=70 skipped_reqs=2",
           "STOA_HITS reqs=5 measured_reqs=3 measured_tokens=300 measured_gpu_hit=30 measured_host_hit=60"]
    run = wc.parse_run(_write(tmp_path, "offload_oracle_x0.35", recs, log), warm=2)
    assert run["hits"] == {"reqs": 3, "tokens": 300, "gpu_hit": 30, "host_hit": 60}
    assert run["offload_stored_gpu_blocks"] == 70


def test_committed_artifact_rederives_and_its_checks_hold():
    d = json.loads(wc.OUT.read_text())
    assert wc.summarize(d) == d["summary"]
    assert all(d["summary"]["checks"].values())
