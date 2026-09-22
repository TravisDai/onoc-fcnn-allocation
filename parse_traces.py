"""Compress the profiler traces into one small CSV.

Each trace file NN{k}_P{period}_B{batch}[.txt] holds, for every core count c = 1..min(n,1000),
100 repeated runs of the per-core computation of one period, with the per-core FLOPs.
Output: traces/compute_traces.csv with one row per (net, period, batch, cores):
  mean and median time over the repetitions, per-core FLOPs, layer sizes.
"""
import csv, io, os, re, sys, zipfile
from collections import defaultdict
import numpy as np

ZIPS = sys.argv[1:]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "compute_traces.csv")
name_re = re.compile(r"(NN\d)_P(\d+)_B(\d+)")
inp_re = re.compile(r"Input parameters:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)")
time_re = re.compile(r"omputation in (\d+) core\(s\) took ([0-9.eE+-]+)s")
flop_re = re.compile(r"FLOP[sS] in (FP|BP) \d+ period is:\s+(\d+)")

rows = []
for zp in ZIPS:
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            m = name_re.search(os.path.basename(info.filename))
            if not m or info.is_dir():
                continue
            net, period, batch = m.group(1), int(m.group(2)), int(m.group(3))
            times = defaultdict(list); flops = {}; params = None; phase = None
            cur = None
            with z.open(info) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    if "Input parameters" in raw:
                        g = inp_re.search(raw); params = g.groups()
                    elif "core(s) took" in raw:
                        g = time_re.search(raw); cur = int(g.group(1)); times[cur].append(float(g.group(2)))
                    elif "period is" in raw:
                        g = flop_re.search(raw); phase = g.group(1); flops[cur] = int(g.group(2))
            mu, a, b, act = int(params[0]), int(params[1]), int(params[2]), params[3]
            if mu != batch:
                print(f"  note: {info.filename} is named B{batch} but contains batch {mu}", flush=True)
                batch = mu          # trust the file content (four NN6 files are misnamed)
            for c in sorted(times):
                t = np.array(times[c])
                rows.append({"net": net, "period": period, "phase": phase, "batch": batch,
                             "p1": a, "p2": b, "act": act, "cores": c, "runs": len(t),
                             "t_mean": t.mean(), "t_median": float(np.median(t)), "t_min": t.min(),
                             "flops": flops.get(c, -1)})
            print(net, period, batch, phase, len(times), flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
print("rows", len(rows))
