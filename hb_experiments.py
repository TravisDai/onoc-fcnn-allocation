"""
Experiments on the Hummingbird-style clustered broadcast network (Section 6.8).

  main      64 clusters x 16 cores, 1 or 4 lanes of 40 Gb/s per hub and direction, 8 dBm lanes,
            all networks, mu in {1,8,16,32}, R and T: exact DP, maximum, best fixed, and the same
            network without the fan-out limit.
  lanes     1, 2, 4, 10 lanes of 40 Gb/s (NN2, NN4, NN6; mu in {1,8,32}).
  rate      the same 40 Gb/s per hub and direction as 4 x 10, 2 x 20 or 1 x 40 Gb/s lanes, with
            0 dBm (comb-class) or 8 dBm (DFB-class) lanes, SDM or WDM.
  clusters  8, 16, 32, 64 clusters of 128, 64, 32, 16 cores at the same optical bandwidth per core.
  dcfg      the main configuration with D_cfg of 1 ns, 10 ns, 100 ns and 1 us (exact DP and maximum).
"""
import sys, time, json, math
from dataclasses import replace
import pandas as pd
import physhb as PH
from hbsim import HCfg, HModel
from onocsim import NETS

SUB = ("NN2", "NN4", "NN6")


def run(net, cfg, tag):
    t0 = time.time()
    M = HModel(net, cfg)
    acl, T = M.dp("FM")
    r = M.full(acl)
    assert abs(r["T"] - T) <= 1e-9 * T
    bf = M.best_fixed()
    row = dict(tag=tag, net=net, C=cfg.C, c=cfg.c, k=cfg.k, B=cfg.B, mux=cfg.mux,
               P_lane_max=cfg.P_lane_max_dbm, mu=cfg.mu, bwd=cfg.bwd, F=cfg.F,
               P_lane_dbm=cfg.lane_power_dbm(), IL_window=float(cfg.pack()[25]),
               acl=json.dumps(acl))
    row.update(r)
    row["T_max"] = M.T(M.max_alloc()); row["T_bf"] = bf[1]; row["u_bf"] = bf[2]
    Mi = HModel(net, replace(cfg, ideal=True))
    ai, Ti = Mi.dp("FM")
    ri = Mi.full(ai)
    row["T_ideal"] = Ti; row["E_ideal"] = ri["E_total"]; row["acl_ideal"] = json.dumps(ai)
    far = cfg.C // 2
    row["IL_ideal"] = PH.il_db(far, far, cfg.mux, cfg.lam_wg, cfg.pitch_mm, cfg.phys)
    row["P_ideal_dbm"] = PH.sens_dbm(cfg.B, cfg.phys) + cfg.phys.margin_db + row["IL_ideal"]
    row["secs"] = time.time() - t0
    print(tag, net, cfg.mu, cfg.bwd, f"C={cfg.C} k={cfg.k} B={cfg.B/1e9:.0f}G {cfg.mux} P={cfg.P_lane_max_dbm} "
          f"F={cfg.F} T={T*1e6:.1f} max={row['T_max']*1e6:.1f} ideal={Ti*1e6:.1f} ({row['secs']:.1f}s)", flush=True)
    return row


def main():
    rows = []
    for net in NETS:
        for mu in (1, 8, 16, 32):
            for bwd in ("R", "T"):
                for k in (1, 4):
                    rows.append(run(net, HCfg(k=k, mu=mu, bwd=bwd), "main"))
        pd.DataFrame(rows).to_csv("results/hb_main.csv", index=False)


def lanes():
    rows = []
    for net in SUB:
        for mu in (1, 8, 32):
            for bwd in ("R", "T"):
                for k in (1, 2, 4, 10):
                    rows.append(run(net, HCfg(k=k, mu=mu, bwd=bwd), "lanes"))
        pd.DataFrame(rows).to_csv("results/hb_lanes.csv", index=False)


def rate():
    rows = []
    for net in SUB:
        for mu in (1, 8, 32):
            for bwd in ("R", "T"):
                for P in (0.0, 8.0):
                    for mux in ("SDM", "WDM"):
                        for B, k in ((10e9, 4), (20e9, 2), (40e9, 1)):
                            rows.append(run(net, HCfg(k=k, B=B, mux=mux, P_lane_max_dbm=P, mu=mu, bwd=bwd), "rate"))
        pd.DataFrame(rows).to_csv("results/hb_rate.csv", index=False)


def clusters():
    rows = []
    for net in SUB:
        for mu in (1, 8, 32):
            for bwd in ("R", "T"):
                for C in (8, 16, 32, 64):
                    c = 1024 // C
                    cfg = HCfg(C=C, c=c, k=64 // C, mu=mu, bwd=bwd, pitch_mm=PH.pitch_for(C),
                               B_cl=4 * c * 3.4e9)
                    rows.append(run(net, cfg, "clusters"))
        pd.DataFrame(rows).to_csv("results/hb_clusters.csv", index=False)


def dcfg():
    rows = []
    for net in NETS:
        for mu in (1, 8, 16, 32):
            for bwd in ("R", "T"):
                for k in (1, 4):
                    for D in (1e-9, 1e-8, 1e-7, 1e-6):
                        M = HModel(net, HCfg(k=k, mu=mu, bwd=bwd, D_cfg=D))
                        acl, T = M.dp("FM")
                        rows.append(dict(net=net, mu=mu, bwd=bwd, k=k, D_cfg=D, T=T,
                                         T_max=M.T(M.max_alloc()), acl=json.dumps(acl)))
            print("dcfg", net, mu, flush=True)
        pd.DataFrame(rows).to_csv("results/hb_dcfg.csv", index=False)


if __name__ == "__main__":
    for part in sys.argv[1:]:
        {"main": main, "lanes": lanes, "rate": rate, "clusters": clusters, "dcfg": dcfg}[part]()
