"""
Runs every experiment reported in the revised manuscript and writes CSV files to results/.

  python experiments.py main      # allocations, baselines, energy, mapping families (Tables 6-7, Figs 6-7)
  python experiments.py dp        # exact chain-DP optima for solver quality (Table 5)
  python experiments.py sens      # sensitivity sweeps (Fig 8)
  python experiments.py enoc      # analytical electrical-mesh reference (Table 8)
"""
from __future__ import annotations
import csv, json, os, sys, time, itertools
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import onocsim as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
NETS = list(S.NETS)
MUS = [1, 8, 16, 32]      # batch sizes with profiler traces
LAMS = [8, 64]
BWDS = ["R", "T"]


def row_for(mod: S.Model, tag, alloc, fam, reuse=-1, extra=None):
    d = mod.full(alloc, fam, reuse)
    Sets = S.core_sets(alloc, mod.cfg.m, fam, reuse)
    sram, _ = S.sram_peak(mod.n, Sets, mod.cfg)
    met = S.schedule_metrics(Sets, mod.cfg.m, d["perT"])
    r = {"method": tag, "family": fam, "reuse": reuse, "alloc": json.dumps(list(map(int, alloc))),
         "T": d["T"], "comp": d["comp"], "comm": d["comm"], "commF": d["commF"], "commB": d["commB"],
         "D_init": d["D_init"], "E_laser": d["E_laser"], "E_tune": d["E_tune"], "E_dyn": d["E_dyn"],
         "E_leak": d["E_leak"], "E_net": d["E_net"], "E_comp": d["E_comp"], "E_total": d["E_total"],
         "flops": d["flops"], "slotsF": d["slotsF"], "slotsB": d["slotsB"], "maxroute": d["maxroute"],
         "maxfan": d["maxfan"], "wire_bytes": d["wireF"] + d["wireB"], "sram_peak": sram,
         "Z": met["Z"], "Rrun": met["Rrun"], "Hmax": met["Hmax"], "cores_used": met["cores_used"],
         "local_frac": S.local_fraction(mod.n, Sets)}
    if extra:
        r.update(extra)
    return r


def main_job(args):
    name, mu, lam, bwd = args
    cfg = S.Cfg(mu=mu, lam=lam, bwd=bwd)
    n = S.NETS[name]
    mod = S.Model(n, cfg, trace_net=name)
    base = {"net": name, "mu": mu, "lam": lam, "bwd": bwd}
    rows = []
    t0 = time.time()
    ri = mod.relaxed()
    rows.append(row_for(mod, "RI", ri, "FM", extra={"relaxed_real": json.dumps(mod.relaxed_real())}))
    for fam in ("FM", "RRM"):
        a, T, sw, ev = mod.coord(ri, fam)
        rows.append(row_for(mod, "CS", a, fam, extra={"sweeps": sw, "evals": ev}))
    a, T, r = mod.orrm_best(ri)
    rows.append(row_for(mod, "CS", a, "ORRM", r))
    a, T, sw, ev = mod.coord(ri, "ORRM", -1)
    rows.append(row_for(mod, "CS_Er", a, "ORRM", -1))
    # energy-optimal allocation (same search, objective = network + compute energy)
    best = None
    for init in (ri, mod.fixed(cfg.m), mod.fixed(1)):
        a, E, sw, ev = mod.coord(init, "FM", obj=1)
        if best is None or E < best[1] - 1e-18:
            best = (a, E)
    rows.append(row_for(mod, "CSE", best[0], "FM"))
    u, E = mod.best_fixed("FM", obj=1)
    rows.append(row_for(mod, "BESTFIXE", mod.fixed(u), "FM", extra={"u": u}))
    for fam in ("FM", "RRM", "ORRM"):
        rows.append(row_for(mod, "MAX", mod.fixed(cfg.m), fam))
        rows.append(row_for(mod, "F200", mod.fixed(200), fam))
        u, T = mod.best_fixed(fam)
        rows.append(row_for(mod, "BESTFIX", mod.fixed(u), fam, extra={"u": u}))
    for r in rows:
        r.update(base)
    return rows, time.time() - t0


def run_main():
    jobs = list(itertools.product(NETS, MUS, LAMS, BWDS))
    path = os.path.join(OUT, "main.csv")
    allrows = []
    with Pool(2) as pool:
        for rows, dt in pool.imap_unordered(main_job, jobs):
            allrows.extend(rows)
            print(rows[0]["net"], rows[0]["mu"], rows[0]["lam"], rows[0]["bwd"], f"{dt:.1f}s", flush=True)
    keys = list(dict.fromkeys(k for r in allrows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)


def run_dp():
    path = os.path.join(OUT, "dp.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["net"], int(r["mu"]), int(r["lam"]), r["bwd"], r["family"]))
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["net", "mu", "lam", "bwd", "family", "alloc", "T", "seconds"])
    for fam in ("FM", "RRM"):
        for name, mu, lam, bwd in itertools.product(NETS, MUS, LAMS, BWDS):
            if (name, mu, lam, bwd, fam) in done:
                continue
            if fam == "RRM" and (mu not in (8, 32) or name not in ("NN1", "NN2", "NN3")):
                continue   # RRM: subset (NN1-NN3, mu in {8,32}) to bound run time
            mod = S.Model(S.NETS[name], S.Cfg(mu=mu, lam=lam, bwd=bwd), trace_net=name)
            t0 = time.time()
            a, T = mod.dp(fam)
            w.writerow([name, mu, lam, bwd, fam, json.dumps(a), T, time.time() - t0]); f.flush()
            print("dp", fam, name, mu, lam, bwd, f"{time.time()-t0:.0f}s", flush=True)
    f.close()


SENS = {
    "D_cfg": [1e-9, 10e-9, 100e-9, 1e-6],
    "C": [1.5e9, 6e9, 24e9],
    "B_lam": [10e9, 40e9],
    "ser2": [2],
}


def sens_job(args):
    name, mu, lam, bwd, key, val = args
    cfg = S.Cfg(mu=mu, lam=lam, bwd=bwd)
    if key == "D_cfg":
        cfg = replace(cfg, D_cfg=val)
    elif key == "C":
        cfg = replace(cfg, C_F=val, C_B=val)
    elif key == "B_lam":
        cfg = replace(cfg, B_lam=val)
    elif key == "ser2":
        cfg = replace(cfg, ser_cyc_override=int(val))
    mod = S.Model(S.NETS[name], cfg, trace_net=None if key == "C" else name)
    ri = mod.relaxed()
    a, T, sw, ev = mod.coord(ri, "FM")
    Tmax = mod.T(mod.fixed(cfg.m)); T200 = mod.T(mod.fixed(200))
    u, Tbf = mod.best_fixed("FM")
    d = mod.full(a)
    return {"net": name, "mu": mu, "lam": lam, "bwd": bwd, "param": key, "value": val,
            "alloc": json.dumps(a), "T": T, "comm_share": d["comm"] / d["T"], "T_max": Tmax,
            "T_200": T200, "T_bestfix": Tbf, "u_bestfix": u, "T_RI": mod.T(ri)}


def run_sens():
    jobs = []
    for name in ["NN2", "NN6"]:
        for mu in [1, 8, 32]:
            for lam in LAMS:
                for bwd in BWDS:
                    for key, vals in SENS.items():
                        for v in vals:
                            jobs.append((name, mu, lam, bwd, key, v))
    rows = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(sens_job, jobs):
            rows.append(r)
    with open(os.path.join(OUT, "sens.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- electrical mesh
BE_GRID = list(np.logspace(9, 13, 33))   # 1 Gb/s .. 10 Tb/s per link and direction
X, Y, T_R = 25, 40, 2


def mesh_T(mod, alloc, B_e, tree):
    return S.mesh_time(mod.n, alloc, mod.cF, mod.cB0, mod.redc, mod.cfg, X, Y, B_e, T_R, tree)[0]


def mesh_coord(mod, init, B_e, tree, max_sweeps=50):
    alloc = list(init)
    best = mesh_T(mod, alloc, B_e, tree)
    for _ in range(max_sweeps):
        improved = False
        for i in range(mod.L):
            cur = alloc[i]; bu, bi = cur, best
            for u in range(1, int(mod.maxdom[i]) + 1):
                if u == cur:
                    continue
                alloc[i] = u
                T = mesh_T(mod, alloc, B_e, tree)
                if T < bi - 1e-15 or (abs(T - bi) <= 1e-15 and u < bu):
                    bi, bu = T, u
            alloc[i] = cur
            if bu != cur and bi < best - 1e-15:
                alloc[i], best, improved = bu, bi, True
        if not improved:
            break
    return alloc, best


def enoc_job(args):
    name, mu, bwd, tree = args
    cfg = S.Cfg(mu=mu, bwd=bwd)
    mod = S.Model(S.NETS[name], cfg, trace_net=name)
    rows = []
    init = mod.fixed(cfg.m)
    for B_e in BE_GRID + [1e18]:
        a, T = mesh_coord(mod, init, B_e, tree)
        init = a
        rows.append({"net": name, "mu": mu, "bwd": bwd, "tree": int(tree), "B_e": B_e,
                     "alloc": json.dumps(a), "T": T})
    # reference link: 128-bit flit per cycle at 3.4 GHz
    Bref = 128 * 3.4e9
    a, T = mesh_coord(mod, mod.fixed(cfg.m), Bref, tree)
    rows.append({"net": name, "mu": mu, "bwd": bwd, "tree": int(tree), "B_e": Bref,
                 "alloc": json.dumps(a), "T": T})
    return rows


def run_enoc():
    jobs = list(itertools.product(NETS, MUS, BWDS, [True, False]))
    rows = []
    with Pool(2) as pool:
        for rs in pool.imap_unordered(enoc_job, jobs):
            rows.extend(rs)
            print("enoc", rs[0]["net"], rs[0]["mu"], rs[0]["bwd"], rs[0]["tree"], flush=True)
    with open(os.path.join(OUT, "enoc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    what = sys.argv[1:] or ["main"]
    for w in what:
        {"main": run_main, "dp": run_dp, "sens": run_sens, "enoc": run_enoc}[w]()
