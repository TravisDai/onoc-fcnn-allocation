"""
hbsim: a Hummingbird-style clustered broadcast optical network for the FCNN workloads
(Section 6.8 of the manuscript).

Organisation (after Lightelligence Hummingbird, Hot Chips 2023)
  * C clusters of c cores; the electronic die is stacked on the photonic die, one optical hub
    per cluster, placed along a crossing-free serpentine (U-shaped) path. Within a cluster the
    hub exchanges data with its cores over an electrical fabric of bandwidth B_cl.
  * Every hub owns k single-writer transmit lanes per direction of the path, at B bit/s each.
    A lane is tapped by the receivers of the next F hubs downstream (fixed taps), so one
    transmission reaches F clusters; F = min(6, link-budget limit) from physhb.py.
  * A multicast whose destinations lie beyond the window is forwarded: the hub at the end of
    each window regenerates the data and transmits it on its own lanes over the next window.
    Relayed data therefore shares the relay hub's lanes with its own traffic. Under R, a relay
    forwards only the rows still needed downstream.
  * Lanes are dedicated, so there is no wavelength slotting and no arbitration. A transition
    lasts D_cfg + max over hubs and directions of (bytes sent / k lanes) serialised at the lane
    flit time + the longest relay-chain latency, or the fabric time if that is longer.
  * ideal=True removes the fan-out limit (one window reaches every destination), to isolate the
    cost of the splitting-loss rule.
Allocation: a_i whole clusters per layer (m_i = min(n_i, c a_i) cores); the chain DP of the
manuscript is exact over a_i = 1..min(C, ceil(n_i/c)).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
import numpy as np
from numba import njit, prange

import physhb as PH
from onocsim import NETS, Cfg, compute_tables, compute_tables_from_traces, flops_per_neuron, d_init


@dataclass(frozen=True)
class HCfg:
    C: int = 64                  # clusters (hubs)
    c: int = 16                  # cores per cluster
    k: float = 10                # lanes per hub per direction
    B: float = 40e9              # bit/s per lane
    mux: str = "SDM"             # 'SDM' (one wavelength per waveguide) or 'WDM'
    lam_wg: int = 16             # wavelengths per waveguide (WDM)
    P_lane_max_dbm: float = 8.0  # laser class: on-chip power available per lane
    pitch_mm: float = 2.5
    f_clk: float = 3.4e9
    flit: int = 16
    D_cfg: float = 10e-9         # per transition (tap and receiver set-up)
    D_eo_cyc: int = 1
    D_oe_cyc: int = 1
    n_group: float = 4.2
    B_cl: float = 64 * 3.4e9     # cluster fabric, bytes/s per direction (4 B/cycle per core)
    mem_bw: float = 25.6e9
    mu: int = 8
    psi: int = 8
    bwd: str = "R"
    ideal: bool = False
    phys: PH.HBPhys = PH.HBPhys()
    eta_laser: float = 0.15
    P_tune: float = 1.097e-3     # per ring on resonance
    P_leak: float = 0.474e-3     # per hub
    E_tx: float = (0.19 + 0.42 + 0.18) * 1e-12
    E_rx: float = (0.19 + 0.18) * 1e-12
    e_fab: float = 0.5e-12
    E_flop: float = 10e-12
    P_fixed_dbm: float = 8.0     # ideal reference: fixed launch power per lane

    @property
    def F(self) -> int:
        return PH.f_max(self.B, self.P_lane_max_dbm, self.mux, self.lam_wg, self.pitch_mm, self.phys)

    @property
    def T_flit(self) -> float:
        return math.ceil(8 * self.flit * self.f_clk / self.B) / self.f_clk

    def lane_power_dbm(self) -> float:
        return PH.lane_power_dbm(self.B, self.F, self.mux, self.lam_wg, self.pitch_mm, self.phys)

    def flat_cfg(self) -> Cfg:
        return Cfg(m=self.C * self.c, lam=64, f_clk=self.f_clk, B_lam=self.B, flit=self.flit,
                   D_cfg=self.D_cfg, mu=self.mu, psi=self.psi, bwd=self.bwd, mem_bw=self.mem_bw,
                   E_flop=self.E_flop)

    def pack(self) -> np.ndarray:
        F = self.F
        if F < 1:
            raise ValueError("link budget cannot reach even one receiver")
        Pl = self.P_fixed_dbm if self.ideal else self.lane_power_dbm()
        return np.array([
            self.C, self.k, self.mu, self.psi, self.flit, self.T_flit,                 # 0-5
            self.D_cfg, (self.D_eo_cyc + self.D_oe_cyc) / self.f_clk,                   # 6-7
            self.pitch_mm * 1e-3 * self.n_group / 299792458.0, self.B_cl,               # 8-9
            1.0 if self.ideal else 0.0, F, self.B, 10 ** (Pl / 10) * 1e-3,              # 10-13
            self.eta_laser, self.P_tune, self.E_tx, self.E_rx, self.e_fab,              # 14-18
            self.P_leak, self.E_flop, 0.0, 1.0 if self.bwd == "T" else 0.0, self.c,     # 19-23
            2.0 if self.mux == "WDM" else 0.0,                                          # 24 rx rings per lane
            PH.il_db(F, F, self.mux, self.lam_wg, self.pitch_mm, self.phys),            # 25
        ], dtype=np.float64)


@njit(cache=True)
def cluster_rows(n, m_cores, c, p):
    """Rows owned by the p-th cluster of an arc when n rows are dealt round-robin to m_cores
    cores that fill the clusters in order (c per cluster). Returns (rows, cores)."""
    lo = p * c
    cnt = m_cores - lo
    if cnt > c:
        cnt = c
    if cnt <= 0:
        return 0, 0
    base = n // m_cores
    rem = n % m_cores
    extra = rem - lo
    if extra < 0:
        extra = 0
    if extra > cnt:
        extra = cnt
    return cnt * base + extra, cnt


@njit(cache=True)
def htransition(s0, a, d0, b, mode, n_src, n_dst, prm, out):
    """One transition from source clusters s0.. (a of them) to destination clusters d0..
    (b of them). mode 0: broadcast of owned rows (FP, T); mode 1: R (per-destination rows).
    out <- time, E_laser, E_tune, E_dyn, E_fab, max chain, segments, relays, T_opt, T_fab,
           max lane load (bytes per hub and direction)."""
    C = int(prm[0]); k = prm[1]; mu = prm[2]; psi = prm[3]; flit = prm[4]; Tfl = prm[5]
    Dcfg = prm[6]; Deoe = prm[7]; thop = prm[8]; Bcl = prm[9]; ideal = prm[10] > 0.5
    F = int(prm[11]); Bl = prm[12]; Pl = prm[13]; eta = prm[14]; Ptune = prm[15]
    Etx = prm[16]; Erx = prm[17]; efab = prm[18]; c = int(prm[23]); rxr = prm[24]
    m_src = n_src if n_src < c * a else c * a
    m_dst = n_dst if n_dst < c * b else c * b
    isdst = np.zeros(C, dtype=np.int64) - 1
    drows = np.zeros(C)
    for q in range(b):
        h = (d0 + q) % C
        r, cn = cluster_rows(n_dst, m_dst, c, q)
        isdst[h] = q; drows[h] = r
    fin = np.zeros(C); fout = np.zeros(C)
    for q in range(b):
        h = (d0 + q) % C
        if mode == 0:
            fout[h] += mu * psi * n_src
        else:
            fout[h] += mu * psi * drows[h]
    txl = np.zeros((2, C))
    dist = np.zeros(C, dtype=np.int64); rw = np.zeros(C)
    Elas = 0.0; Etun = 0.0; Edyn = 0.0
    maxlat = 0.0; maxJ = 0; nseg = 0; nrel = 0; traffic = False
    for p in range(a):
        kh = (s0 + p) % C
        r_src, cn_src = cluster_rows(n_src, m_src, c, p)
        if mode == 0:
            fin[kh] += mu * psi * r_src
        else:
            fin[kh] += cn_src * mu * psi * n_dst
        bb = mu * psi * r_src
        for dirn in range(2):
            R = 0
            for dd in range(1, C):
                if dirn == 0:
                    h = (kh + dd) % C
                    if dd > C - dd:
                        break
                else:
                    h = (kh - dd) % C
                    if dd >= C - dd:
                        break
                if isdst[h] >= 0:
                    dist[R] = dd; rw[R] = drows[h]; R += 1
            if R == 0:
                continue
            traffic = True
            D = dist[R - 1]
            Fw = D if ideal else F
            J = (D + Fw - 1) // Fw
            lat = 0.0
            for j in range(J):
                st = j * Fw
                en = (j + 1) * Fw
                if en > D:
                    en = D
                pos = (kh + st) % C if dirn == 0 else (kh - st) % C
                if mode == 0:
                    by = bb
                else:
                    by = 0.0
                    for t in range(R):
                        if dist[t] > st:
                            by += rw[t]
                    by *= mu * psi
                txl[dirn, pos] += by
                nd = 0; own = 0.0; relay_dest = False
                for t in range(R):
                    if dist[t] > st and dist[t] <= en:
                        nd += 1; own += rw[t]
                        if dist[t] == en:
                            relay_dest = True
                relay = j < J - 1
                if mode == 0:
                    nrx = nd + (1 if (relay and not relay_dest) else 0)
                    rxb = 8.0 * by * nrx
                else:
                    nxt = 0.0
                    if relay:
                        for t in range(R):
                            if dist[t] > en:
                                nxt += rw[t]
                    nrx = nd + (1 if (relay and not relay_dest) else 0)
                    rxb = 8.0 * mu * psi * (own + nxt)
                bits = 8.0 * by
                Edyn += bits * Etx + rxb * Erx
                Elas += bits / Bl * Pl / eta
                Etun += bits / Bl * Ptune * (1.0 + rxr * nrx)
                lat += thop * (en - st) + Deoe
                nseg += 1
                if relay:
                    nrel += 1
            lat += (J - 1) * Tfl
            if lat > maxlat:
                maxlat = lat
            if J > maxJ:
                maxJ = J
    mload = 0.0
    for dirn in range(2):
        for h in range(C):
            if txl[dirn, h] > mload:
                mload = txl[dirn, h]
    Topt = 0.0
    if traffic:
        Topt = Dcfg + math.ceil(mload / (flit * k) - 1e-9) * Tfl + maxlat
    Tfab = 0.0; fb = 0.0
    for h in range(C):
        x = fin[h] if fin[h] > fout[h] else fout[h]
        if x / Bcl > Tfab:
            Tfab = x / Bcl
        fb += fin[h] + fout[h]
    out[0] = Topt if Topt > Tfab else Tfab
    out[1] = Elas; out[2] = Etun; out[3] = Edyn; out[4] = 8.0 * fb * efab
    out[5] = maxJ; out[6] = nseg; out[7] = nrel; out[8] = Topt; out[9] = Tfab; out[10] = mload


@njit(cache=True)
def hstarts(acl, C, fam):
    L = acl.shape[0]
    st = np.zeros(L, dtype=np.int64)
    if fam == 1:
        for i in range(1, L):
            st[i] = (st[i - 1] + acl[i - 1]) % C
    return st


@njit(cache=True)
def hevaluate(n, acl, fam, cF, cB0, prm, fl0, flr, res):
    """res <- T, comp, comm, E_laser, E_tune, E_dyn, E_fab, E_leak, E_comp, E_total, max chain,
    segments, relays, commF, commB, fabric-bound transitions."""
    L = acl.shape[0]
    C = int(prm[0]); c = int(prm[23]); bwdT = prm[22] > 0.5
    st = hstarts(acl, C, fam)
    out = np.zeros(11)
    comp = 0.0
    for i in range(L):
        mi = n[i + 1] if n[i + 1] < c * acl[i] else c * acl[i]
        comp += cF[i, mi] + cB0[i, mi]
    cf = 0.0; cb = 0.0; El = 0.0; Et = 0.0; Ed = 0.0; Ef = 0.0; mJ = 0.0; ns = 0.0; nr = 0.0; fbd = 0.0
    for i in range(L - 1):
        htransition(st[i], acl[i], st[i + 1], acl[i + 1], 0, n[i + 1], n[i + 2], prm, out)
        cf += out[0]; El += out[1]; Et += out[2]; Ed += out[3]; Ef += out[4]
        mJ = max(mJ, out[5]); ns += out[6]; nr += out[7]
        if out[9] > out[8]:
            fbd += 1
    for i in range(1, L):
        htransition(st[i], acl[i], st[i - 1], acl[i - 1], 0 if bwdT else 1, n[i + 1], n[i], prm, out)
        cb += out[0]; El += out[1]; Et += out[2]; Ed += out[3]; Ef += out[4]
        mJ = max(mJ, out[5]); ns += out[6]; nr += out[7]
        if out[9] > out[8]:
            fbd += 1
    T = prm[21] + comp + cf + cb
    fl = 0.0
    for i in range(L):
        fl += fl0[i]
        if i < L - 1:
            mi1 = n[i + 2] if n[i + 2] < c * acl[i + 1] else c * acl[i + 1]
            fl += flr[i] * (mi1 - 1)
    Eleak = prm[19] * C * T
    Ecomp = fl * prm[20]
    res[0] = T; res[1] = comp; res[2] = cf + cb; res[3] = El; res[4] = Et; res[5] = Ed; res[6] = Ef
    res[7] = Eleak; res[8] = Ecomp; res[9] = El + Et + Ed + Ef + Eleak + Ecomp
    res[10] = mJ; res[11] = ns; res[12] = nr; res[13] = cf; res[14] = cb; res[15] = fbd
    return T


@njit(cache=True, parallel=True)
def hpair_matrix(li, n, fam, prm, A, Bm):
    C = int(prm[0]); bwdT = prm[22] > 0.5
    P = np.full((A + 1, Bm + 1), np.inf)
    for a in prange(1, A + 1):
        out = np.zeros(11)
        for b in range(1, Bm + 1):
            s1 = 0 if fam == 0 else a % C
            htransition(0, a, s1, b, 0, n[li + 1], n[li + 2], prm, out)
            t = out[0]
            htransition(s1, b, 0, a, 0 if bwdT else 1, n[li + 2], n[li + 1], prm, out)
            P[a, b] = t + out[0]
    return P


FAMH = {"FM": 0, "RRM": 1}
KEYS = ["T", "comp", "comm", "E_laser", "E_tune", "E_dyn", "E_fab", "E_leak", "E_comp", "E_total",
        "max_chain", "segments", "relays", "commF", "commB", "fabric_bound"]


class HModel:
    def __init__(self, net: str, cfg: HCfg, n=None, analytic=False):
        self.net = net; self.n = list(NETS[net] if n is None else n); self.cfg = cfg
        self.na = np.asarray(self.n, dtype=np.int64); self.L = len(self.n) - 1
        fc = cfg.flat_cfg()
        if analytic:
            self.cF, self.cB0, _ = compute_tables(self.n, fc)
        else:
            self.cF, self.cB0, _ = compute_tables_from_traces(net, self.n, fc)
        self.prm = cfg.pack(); self.prm[21] = d_init(self.n, fc)
        self.maxdom = [min(cfg.C, -(-self.n[i + 1] // cfg.c)) for i in range(self.L)]
        self.fl0 = np.zeros(self.L); self.flr = np.zeros(self.L)
        for i in range(1, self.L + 1):
            fF, fB, red = flops_per_neuron(self.n, i, fc)
            self.fl0[i - 1] = self.n[i] * (fF + fB); self.flr[i - 1] = self.n[i] * red

    def m_of(self, acl):
        return [min(self.n[i + 1], self.cfg.c * acl[i]) for i in range(self.L)]

    def full(self, acl, fam="FM"):
        res = np.zeros(16)
        hevaluate(self.na, np.asarray(acl, dtype=np.int64), FAMH[fam], self.cF, self.cB0, self.prm,
                  self.fl0, self.flr, res)
        return dict(zip(KEYS, res.tolist()))

    def T(self, acl, fam="FM"):
        return self.full(acl, fam)["T"]

    def dp(self, fam="FM"):
        f = FAMH[fam]; L = self.L; md = self.maxdom; c = self.cfg.c
        node = [np.array([np.inf] + [self.cF[i, min(self.n[i + 1], c * a)] +
                                     self.cB0[i, min(self.n[i + 1], c * a)]
                                     for a in range(1, md[i] + 1)]) for i in range(L)]
        V = node[0].copy(); back = []
        for li in range(L - 1):
            P = hpair_matrix(li, self.na, f, self.prm, md[li], md[li + 1])
            tot = V[:, None] + P
            arg = np.argmin(tot[1:, :], axis=0) + 1
            Vn = tot[arg, np.arange(md[li + 1] + 1)] + node[li + 1]
            Vn[0] = np.inf
            back.append(arg); V = Vn
        last = int(np.argmin(V[1:]) + 1)
        acl = [last]
        for arg in reversed(back):
            acl.append(int(arg[acl[-1]]))
        acl.reverse()
        return acl, float(V[last] + self.prm[21])

    def max_alloc(self):
        return list(self.maxdom)

    def best_fixed(self, fam="FM"):
        best = None
        for u in range(1, self.cfg.C + 1):
            acl = [min(u, d) for d in self.maxdom]
            T = self.T(acl, fam)
            if best is None or T < best[1] - 1e-15:
                best = (acl, T, u)
        return best
