"""vLLM 0.11 OffloadingConnector with a per-request store skip (C1 probe).

A request whose kv_transfer_params carry {"stoa.skip_store": true} is never offloaded: before the stock
`_get_reqs_to_store` runs, its next-to-store index is advanced to the blocks it has computed, so the stock
code sees nothing new to store for it. Everything else (loads, LRU, async transfers) is the stock connector.
Load with: kv_connector="SkipStoreOffloadingConnector", kv_connector_module_path="skipstore_connector".
Instrumentation only (no behaviour change): blocks handed to the offloader (STOA_SKIPSTORE) and, per request, the
GPU-local and host-tier hit tokens at its last lookup (STOA_HITS; the first STOA_WARMUP_REQS requests seen are warm-up).
"""
import os
from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector as oc
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector, OffloadingConnectorScheduler)

_stock = OffloadingConnectorScheduler._get_reqs_to_store
_count = {"calls": 0, "stored_gpu_blocks": 0, "store_jobs": 0, "skipped_reqs": set()}
_stock_match = OffloadingConnectorScheduler.get_num_new_matched_tokens
_hits = {}  # req_id -> [arrival order, num_tokens, gpu-local hit tokens, host-tier hit tokens]


def get_num_new_matched_tokens(self, request, num_computed_tokens):
    ext, load_async = _stock_match(self, request, num_computed_tokens)
    try:
        prev = _hits.get(request.request_id)
        _hits[request.request_id] = [prev[0] if prev else len(_hits), request.num_tokens, num_computed_tokens, ext or 0]
    except Exception:
        pass
    return ext, load_async


def _hit_line():
    warm = int(os.environ.get("STOA_WARMUP_REQS", "400"))
    m = [v for v in _hits.values() if v[0] >= warm]
    return (f"STOA_HITS reqs={len(_hits)} measured_reqs={len(m)} measured_tokens={sum(v[1] for v in m)} "
            f"measured_gpu_hit={sum(v[2] for v in m)} measured_host_hit={sum(v[3] for v in m)}")


def _get_reqs_to_store(self, scheduler_output):
    for req_id, _new_blocks, _preempted in oc.yield_req_data(scheduler_output):
        req = self._requests.get(req_id)
        params = getattr(req, "kv_transfer_params", None) or {}
        if req is None or not params.get("stoa.skip_store"):
            continue
        total = req.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
        self._next_stored_block_idx[req_id] = total // self.offloaded_block_size
        _count["skipped_reqs"].add(req_id)
    out = _stock(self, scheduler_output)
    # evidence that the skip takes effect: GPU blocks actually handed to the offloading worker
    _count["calls"] += 1
    _count["store_jobs"] += len(out)
    _count["stored_gpu_blocks"] += sum(len(src.block_ids) for src, _dst in out.values())
    if _count["calls"] % 100 == 0:
        print(f"STOA_SKIPSTORE calls={_count['calls']} store_jobs={_count['store_jobs']} "
              f"stored_gpu_blocks={_count['stored_gpu_blocks']} skipped_reqs={len(_count['skipped_reqs'])}",
              flush=True)
        print(_hit_line(), flush=True)
    return out


OffloadingConnectorScheduler._get_reqs_to_store = _get_reqs_to_store
OffloadingConnectorScheduler.get_num_new_matched_tokens = get_num_new_matched_tokens


class SkipStoreOffloadingConnector(OffloadingConnector):
    pass
