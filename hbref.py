"""
Independent, slow reference for one transition of the Hummingbird-style network
(hbsim.htransition), written with explicit lists and dictionaries and no shared code.
It also returns the relay chains so the tests can check coverage and the fan-out limit.
"""
from __future__ import annotations
import math


def split_rows(n, a, c):
    m = min(n, c * a)
    per_core = [n // m + (1 if j < n % m else 0) for j in range(m)]
    return ([sum(per_core[p * c:(p + 1) * c]) for p in range(a)],
            [len(per_core[p * c:(p + 1) * c]) for p in range(a)])


def transition_ref(s0, a, d0, b, mode, n_src, n_dst, P):
    C, k, c, F = P["C"], P["k"], P["c"], P["F"]
    mu, psi = P["mu"], P["psi"]
    srows, scores = split_rows(n_src, a, c)
    drows_l, _ = split_rows(n_dst, b, c)
    dst = {(d0 + q) % C: drows_l[q] for q in range(b)}
    fab_in = {h: 0.0 for h in range(C)}; fab_out = {h: 0.0 for h in range(C)}
    for h, r in dst.items():
        fab_out[h] += mu * psi * (n_src if mode == 0 else r)
    load = {}
    E = dict(laser=0.0, tune=0.0, dyn=0.0)
    chains = []
    lat_max = 0.0
    for p in range(a):
        src = (s0 + p) % C
        fab_in[src] += mu * psi * (srows[p] if mode == 0 else scores[p] * n_dst)
        cw = sorted(((h - src) % C, r) for h, r in dst.items()
                    if h != src and (h - src) % C <= C - (h - src) % C)
        ccw = sorted(((src - h) % C, r) for h, r in dst.items()
                     if h != src and (h - src) % C > C - (h - src) % C)
        for dirn, recv in ((0, cw), (1, ccw)):
            if not recv:
                continue
            far = recv[-1][0]
            win = far if P["ideal"] else F
            starts = list(range(0, far, win))
            chain = []
            lat = 0.0
            for j, st in enumerate(starts):
                en = min(st + win, far)
                tx = (src + st) % C if dirn == 0 else (src - st) % C
                if mode == 0:
                    nbytes = mu * psi * srows[p]
                else:
                    nbytes = mu * psi * sum(r for d, r in recv if d > st)
                load[(dirn, tx)] = load.get((dirn, tx), 0.0) + nbytes
                inside = [(d, r) for d, r in recv if st < d <= en]
                is_relay = j < len(starts) - 1
                relay_is_dest = any(d == en for d, _ in inside)
                n_rx = len(inside) + (1 if is_relay and not relay_is_dest else 0)
                if mode == 0:
                    rx_bits = 8 * nbytes * n_rx
                else:
                    fwd = sum(r for d, r in recv if d > en) if is_relay else 0.0
                    rx_bits = 8 * mu * psi * (sum(r for _, r in inside) + fwd)
                E["dyn"] += 8 * nbytes * P["Etx"] + rx_bits * P["Erx"]
                E["laser"] += 8 * nbytes / P["B"] * P["Pl"] / P["eta"]
                E["tune"] += 8 * nbytes / P["B"] * P["Ptune"] * (1 + P["rxr"] * n_rx)
                lat += P["thop"] * (en - st) + P["Deoe"]
                chain.append(dict(tx=tx, dirn=dirn, start=st, end=en, receivers=[d for d, _ in inside],
                                  relay=is_relay))
            lat += (len(starts) - 1) * P["Tfl"]
            lat_max = max(lat_max, lat)
            chains.append(dict(src=src, dirn=dirn, far=far, segs=chain,
                               wanted=[d for d, _ in recv]))
    t_opt = 0.0
    if load:
        t_opt = P["Dcfg"] + math.ceil(max(load.values()) / (P["flit"] * k) - 1e-9) * P["Tfl"] + lat_max
    t_fab = max(max(fab_in[h], fab_out[h]) for h in range(C)) / P["Bcl"]
    return dict(time=max(t_opt, t_fab), E_laser=E["laser"], E_tune=E["tune"], E_dyn=E["dyn"],
                E_fab=8 * P["efab"] * (sum(fab_in.values()) + sum(fab_out.values())),
                chains=chains, t_opt=t_opt, t_fab=t_fab,
                max_load=max(load.values()) if load else 0.0)


def params(prm):
    return dict(C=int(prm[0]), k=prm[1], mu=prm[2], psi=prm[3], flit=prm[4], Tfl=prm[5],
                Dcfg=prm[6], Deoe=prm[7], thop=prm[8], Bcl=prm[9], ideal=prm[10] > 0.5,
                F=int(prm[11]), B=prm[12], Pl=prm[13], eta=prm[14], Ptune=prm[15], Etx=prm[16],
                Erx=prm[17], efab=prm[18], c=int(prm[23]), rxr=prm[24])
