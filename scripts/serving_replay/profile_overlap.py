"""Overlap of HiCache copy kernels with prefill kernels in the nsys captures of profile_sgl.sh (notebook BM).

Reads prof/{write_through,write_through_selective}.sqlite (nsys export) and writes prof/overlap.json: the copy
kernels (hicache_transfer_all_layer) in the window, and how many FlashInfer prefill kernels overlapped one and
how long they took against those that ran alone.
"""
import sqlite3, sys, bisect
R = {}
for p in ("write_through", "write_through_selective"):
    db = sqlite3.connect(f"prof/{p}.sqlite")
    names = dict(db.execute("select id, value from StringIds"))
    ks = db.execute("select start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    xfer = [(s, e) for s, e, sn, dn in ks if "hicache_transfer_all_layer" in names.get(dn, "")]
    pref = [(s, e) for s, e, sn, dn in ks if "BatchPrefillWithPagedKVCacheKernel" in names.get(dn, "")]
    xs = [s for s, _ in xfer]
    def overlaps(s, e):
        i = bisect.bisect_left(xs, e)
        return any(xs_ < e and xe > s for xs_, xe in xfer[max(0, i - 50):i])
    ov = [e - s for s, e in pref if overlaps(s, e)]
    no = [e - s for s, e in pref if not overlaps(s, e)]
    span = (ks[-1][1] - ks[0][0]) / 1e9
    busy = sum(e - s for s, e in xfer) / 1e9
    import json; R[p]={"window_s": round(span,1), "transfer_kernels": len(xfer), "transfer_busy_s": round(busy,2), "prefill_kernels": len(pref), "prefill_overlapped": len(ov), "overlapped_mean_ms": round(sum(ov)/max(1,len(ov))/1e6,3), "alone_mean_ms": round(sum(no)/max(1,len(no))/1e6,3)}
    print(p, f"window {span:.1f}s transfer kernels {len(xfer)} busy {busy:.2f}s ({100*busy/span:.1f}% of window); "
          f"prefill kernels {len(pref)}: overlapped {len(ov)} mean {sum(ov)/max(1,len(ov))/1e6:.2f} ms, "
          f"alone {len(no)} mean {sum(no)/max(1,len(no))/1e6:.2f} ms")
open("prof/overlap.json","w").write(__import__("json").dumps(R, indent=1))
