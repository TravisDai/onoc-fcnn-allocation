"""Robustness check on ring size (Section 6.6): m in {64, 256} with unchanged per-core parameters."""
import csv, json, os, itertools
from dataclasses import replace
import onocsim as S

rows = []
for m, name, mu, lam, bwd in itertools.product([64, 256, 1000], ["NN2", "NN6"], [8, 32], [8, 64], ["R", "T"]):
    if lam > m:
        continue
    cfg = S.Cfg(m=m, mu=mu, lam=lam, bwd=bwd)
    mod = S.Model(S.NETS[name], cfg, trace_net=name)
    a, T, _, _ = mod.coord(mod.relaxed(), "FM")
    Tmax = mod.T(mod.fixed(m)); u, Tbf = mod.best_fixed("FM")
    rows.append({"m": m, "net": name, "mu": mu, "lam": lam, "bwd": bwd, "alloc": json.dumps(a),
                 "T": T, "T_max": Tmax, "T_bestfix": Tbf, "u_bestfix": u})
with open(os.path.join("results", "ringsize.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows))
