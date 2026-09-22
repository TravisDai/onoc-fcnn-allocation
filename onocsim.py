"""
onocsim: schedule-level simulator for mini-batch FCNN training on a wavelength-slotted
ring optical network-on-chip (ONoC), with an analytical electrical-mesh reference.

This file implements exactly the equations of the revised manuscript
(Sections 3-4). `refmodel.py` is an independent, slower, dictionary-based
implementation of the same equations; `test_onocsim.py` checks that the two agree.

Units: seconds, bytes, joules, watts.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Sequence, Tuple
import numpy as np
from numba import njit, prange

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    # architecture
    m: int = 1000
    lam: int = 64
    f_clk: float = 3.4e9
    C_F: float = 6e9             # sustained FLOP/s per core, forward kernels
    C_B: float = 6e9             # sustained FLOP/s per core, backward kernels
    B_lam: float = 40e9          # bit/s per wavelength
    flit: int = 16               # bytes
    ser_cyc_override: int = 0    # 0 => physical ceil(8*flit*f_clk/B_lam) cycles per flit
    D_cfg: float = 10e-9         # ring-filter/modulator (re)configuration per slot
    D_eo_cyc: int = 1
    D_oe_cyc: int = 1
    hop_mm: float = 0.63         # physical length of one logical hop
    n_group: float = 4.2         # group index of the waveguide
    mem_bw: float = 25.6e9       # bytes/s, main memory (input mini-batch load)
    # workload
    mu: int = 8
    psi: int = 8                 # bytes per scalar (double precision, as in the profiled code)
    bwd: str = "R"               # 'R' partial-error reduction, 'T' replicated transpose broadcast
    lower_order: bool = True
    # energy (network)
    P_laser_opt: float = 0.645e-3  # optical launch power per active wavelength (W)
    eta_laser: float = 0.30
    P_tune: float = 1.097e-3       # per tuned microring (W)
    P_leak: float = 0.474e-3       # per router (W), whole iteration
    E_tx: float = (0.19 + 0.42 + 0.18) * 1e-12   # serdes + modulator + logic (J/bit)
    E_rx: float = (0.19 + 0.18) * 1e-12          # deserialiser + logic per receiver (J/bit)
    E_flop: float = 10e-12         # compute energy per FLOP incl. operand access (J)
    # SRAM
    sram_cap: float = float("inf")  # bytes per core (inf => unconstrained)

    @property
    def T_flit_cyc(self) -> int:
        if self.ser_cyc_override:
            return self.ser_cyc_override
        return math.ceil(8 * self.flit * self.f_clk / self.B_lam)

    @property
    def t_hop(self) -> float:
        return self.hop_mm * 1e-3 * self.n_group / 299792458.0

    def pack(self) -> np.ndarray:
        return np.array([
            self.m, self.lam, self.mu, self.psi, self.f_clk, self.flit,
            self.T_flit_cyc / self.f_clk,                          # 6 seconds per flit
            self.D_cfg + (self.D_eo_cyc + self.D_oe_cyc) / self.f_clk,  # 7 D_fix
            self.t_hop,                                            # 8
            self.P_laser_opt / self.eta_laser,                     # 9 wall-plug per wavelength
            self.P_tune, self.E_tx, self.E_rx, self.P_leak,        # 10..13
            0.0,                                                   # 14 D_init (set per network)
            1.0 if self.bwd == "T" else 0.0,                       # 15
            self.E_flop,                                           # 16
        ], dtype=np.float64)


FAM = {"FM": 0, "RRM": 1, "ORRM": 2}
NETS: Dict[str, List[int]] = {
    "NN1": [784, 1000, 500, 10],
    "NN2": [784, 1500, 784, 1000, 500, 10],
    "NN3": [784, 2000, 1500, 784, 1000, 500, 10],
    "NN4": [784, 2500, 2000, 1500, 784, 1000, 500, 10],
    "NN5": [1024, 4000, 1000, 4000, 10],
    "NN6": [1024, 4000, 1000, 4000, 1000, 4000, 1000, 4000, 10],
}


# --------------------------------------------------------------------------
# Computation model (precomputed per layer and per core count)
# --------------------------------------------------------------------------
def flops_per_neuron(n: Sequence[int], i: int, cfg: Cfg) -> Tuple[float, float, float]:
    """(forward, backward-without-reduction, reduction coefficient) FLOPs per owned neuron of
    layer i. The reduction term (R only) is mu*(m_{i+1}-1) and is returned as the coefficient mu."""
    L = len(n) - 1
    mu = cfg.mu
    aF = 2 * mu * n[i - 1]
    hF = 2 * mu if cfg.lower_order else 0
    if cfg.bwd == "R":
        aB = 2 * mu * n[i - 1] + (2 * mu * n[i - 1] if i > 1 else 0)
        red = mu if i < L else 0
        hB = (3 * mu + 2 * (n[i - 1] + 1)) if cfg.lower_order else 0
    else:
        aB = 2 * mu * n[i - 1] + (4 * mu * n[i + 1] if i < L else 0)
        red = 0
        hB = ((3 * mu + 2 * (n[i - 1] + 1)) + (2 * n[i + 1] if i < L else 0)) if cfg.lower_order else 0
    return aF + hF, aB + hB, red


def compute_tables(n: Sequence[int], cfg: Cfg, trace=None):
    """cF[i-1, a], cB0[i-1, a], redc[i-1, a] for a = 1..m (index a).
    If `trace` is given it must be a callable trace(i, phase, q) -> seconds for one core owning q
    neurons of layer i (phase 'F' or 'B'); it then replaces the FLOP/throughput model and the
    reduction term must be included in the trace (redc is set to 0)."""
    L = len(n) - 1
    M = cfg.m
    cF = np.zeros((L, M + 1)); cB0 = np.zeros((L, M + 1)); redc = np.zeros((L, M + 1))
    a = np.arange(1, M + 1)
    for i in range(1, L + 1):
        q = -(-n[i] // a)
        if trace is None:
            fF, fB, red = flops_per_neuron(n, i, cfg)
            cF[i - 1, 1:] = fF * q / cfg.C_F
            cB0[i - 1, 1:] = fB * q / cfg.C_B
            redc[i - 1, 1:] = red * q / cfg.C_B
        else:
            cF[i - 1, 1:] = [trace(i, "F", int(x)) for x in q]
            cB0[i - 1, 1:] = [trace(i, "B", int(x)) for x in q]
    return cF, cB0, redc


_TRACE_CACHE = {}


def load_traces(path=None):
    """Profiler traces compressed by parse_traces.py (one row per net, period, batch, cores)."""
    import os, pandas as pd
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "compute_traces.csv")
    if path not in _TRACE_CACHE:
        _TRACE_CACHE[path] = pd.read_csv(path)
    return _TRACE_CACHE[path]


def _isotonic(y, w):
    """Weighted pool-adjacent-violators: the non-decreasing sequence closest to y."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
    out = []
    for v, c in zip(vals, cnt):
        out += [v] * c
    return out


def per_load_times(df, net, period, mu, n_layer, monotone=True):
    """Mean per-core time for each per-core load q, aggregating all traced core counts k with
    ceil(n/k) = q (weighted by repetitions). With monotone=True the per-load means are made
    non-decreasing in q by weighted isotonic regression, so that isolated measurement
    artefacts (a larger load timed faster than a smaller one) cannot be exploited."""
    x = df[(df.net == net) & (df.period == period) & (df.batch == mu)]
    if len(x) == 0:
        raise KeyError((net, period, mu))
    q = -(-n_layer // x.cores.values)
    w = x.runs.values * 1.0
    t = x.t_mean.values
    qs = np.unique(q)
    means = [float(np.sum(t[q == qq] * w[q == qq]) / np.sum(w[q == qq])) for qq in qs]
    wts = [float(np.sum(w[q == qq])) for qq in qs]
    if monotone:
        means = _isotonic(means, wts)
    return {int(qq): m for qq, m in zip(qs, means)}


def compute_tables_from_traces(net, n: Sequence[int], cfg: Cfg, df=None):
    """Trace-driven computation (Section 6.1).
    FP: the profiled kernel is the forward product of the layer, used unchanged.
    BP: the profiled kernel forms the weight gradient (2*mu*n_{i-1} FLOPs per neuron). The
    modelled backward work differs by collective, so the measured time is scaled by the ratio
    of modelled to profiled leading-order FLOPs, i.e. the extra products run at the efficiency
    measured for the traced kernel at the same per-core load. The R accumulation term uses the
    same measured efficiency."""
    df = load_traces() if df is None else df
    L = len(n) - 1
    M = cfg.m
    mu = cfg.mu
    cF = np.zeros((L, M + 1)); cB0 = np.zeros((L, M + 1)); redc = np.zeros((L, M + 1))
    a = np.arange(1, M + 1)
    for i in range(1, L + 1):
        q = -(-n[i] // a)
        tF = per_load_times(df, net, i, mu, n[i])
        tB = per_load_times(df, net, 2 * L - i + 1, mu, n[i])
        prof = 2.0 * mu * n[i - 1]
        if cfg.bwd == "R":
            ratio = (2.0 * mu * n[i - 1] * (2 if i > 1 else 1)) / prof
        else:
            ratio = (2.0 * mu * n[i - 1] + (4.0 * mu * n[i + 1] if i < L else 0.0)) / prof
        for idx, qq in enumerate(q):
            qq = int(qq)
            if qq not in tF or qq not in tB:
                cF[i - 1, idx + 1] = np.inf; cB0[i - 1, idx + 1] = np.inf
                continue
            cF[i - 1, idx + 1] = tF[qq]
            cB0[i - 1, idx + 1] = tB[qq] * ratio
            if cfg.bwd == "R" and i < L:
                c_eff = prof * qq / tB[qq]          # measured FLOP/s of the traced kernel at load q
                redc[i - 1, idx + 1] = mu * qq / c_eff
    return cF, cB0, redc


def total_flops(n: Sequence[int], alloc: Sequence[int], cfg: Cfg) -> float:
    L = len(n) - 1
    tot = 0.0
    for i in range(1, L + 1):
        fF, fB, red = flops_per_neuron(n, i, cfg)
        tot += n[i] * (fF + fB + (red * (alloc[i] - 1) if i < L else 0))
    return tot


# --------------------------------------------------------------------------
# Numba kernels: ring ONoC
# --------------------------------------------------------------------------
@njit(cache=True)
def _dring(x, m):
    x = x % m
    return x if x <= m - x else m - x


@njit(cache=True)
def _arc_route(k, d0, b, m):
    off = (k - d0) % m
    last = b - 1
    h = m // 2
    a1 = (off + h) % m
    a2 = (off + (m + 1) // 2) % m
    if a1 <= last or a2 <= last:
        return h
    r1 = _dring(off, m)            # distance to arc start (offset 0)
    r2 = _dring(off - last, m)     # distance to arc end
    return r1 if r1 > r2 else r2


@njit(cache=True)
def transition(m, s0, a, d0, b, mode, n_src, n_dst, prm, out):
    """One slotted multicast transition between contiguous arcs.
    mode 0: payload = mu*psi*q_src   (activation broadcast, or error broadcast for 'T')
    mode 1: payload = mu*psi*(n_dst - own)   (segmented partial-error reduction, 'R')
    out[:] <- time, slots, E_laser, E_tune, E_dyn, maxroute, maxfan, wire_bytes, emitters"""
    lam = int(prm[1]); mu = prm[2]; psi = prm[3]; flit = prm[5]; Tfl = prm[6]
    Dfix = prm[7]; thop = prm[8]; Plas = prm[9]; Ptune = prm[10]; Etx = prm[11]; Erx = prm[12]
    bs = n_src // a; rs = n_src % a
    bd = n_dst // b; rd = n_dst % b
    time = 0.0; slots = 0; Elas = 0.0; Etun = 0.0; txb = 0.0; rxb = 0.0
    maxroute = 0; maxfan = 0; wire = 0.0; emit = 0
    sc = 0; smaxb = 0.0; smaxr = 0; sfan = 0
    for p in range(a):
        k = (s0 + p) % m
        off = (k - d0) % m
        ind = 1 if off < b else 0
        fan = b - ind
        if fan == 0:
            continue
        if mode == 0:
            cnt = bs + (1 if p < rs else 0)
        else:
            own = (bd + (1 if off < rd else 0)) if ind == 1 else 0
            cnt = n_dst - own
        byt = mu * psi * cnt
        r = _arc_route(k, d0, b, m)
        sc += 1; emit += 1
        if byt > smaxb:
            smaxb = byt
        if r > smaxr:
            smaxr = r
        sfan += fan
        txb += 8.0 * byt; wire += byt
        if mode == 0:
            rxb += 8.0 * byt * fan      # every receiver consumes the broadcast
        else:
            rxb += 8.0 * byt            # each segment is consumed by its owner only
        if fan > maxfan:
            maxfan = fan
        if sc == lam:
            t = Dfix + thop * smaxr + math.ceil(smaxb / flit) * Tfl
            time += t; slots += 1
            Elas += t * sc * Plas; Etun += t * Ptune * (sc + sfan)
            if smaxr > maxroute:
                maxroute = smaxr
            sc = 0; smaxb = 0.0; smaxr = 0; sfan = 0
    if sc > 0:
        t = Dfix + thop * smaxr + math.ceil(smaxb / flit) * Tfl
        time += t; slots += 1
        Elas += t * sc * Plas; Etun += t * Ptune * (sc + sfan)
        if smaxr > maxroute:
            maxroute = smaxr
    out[0] = time; out[1] = slots; out[2] = Elas; out[3] = Etun
    out[4] = txb * Etx + rxb * Erx; out[5] = maxroute; out[6] = maxfan; out[7] = wire; out[8] = emit


@njit(cache=True)
def starts_of(alloc, m, fam, reuse):
    L = alloc.shape[0]
    st = np.zeros(L, dtype=np.int64)
    if fam == 1:
        for i in range(1, L):
            st[i] = (st[i - 1] + alloc[i - 1]) % m
    elif fam == 2:
        tot = 0
        for i in range(L):
            tot += alloc[i]
        if reuse >= 0:
            target = reuse
        elif tot <= m or L == 1:
            target = 0
        else:
            # round half to even, as Python's round()
            target = int(np.round((tot - m) / (L - 1)))
        rp = 0
        for i in range(1, L):
            r = target
            if alloc[i - 1] - rp < r:
                r = alloc[i - 1] - rp
            if alloc[i] < r:
                r = alloc[i]
            if r < 0:
                r = 0
            st[i] = (st[i - 1] + alloc[i - 1] - r) % m
            rp = r
    return st


@njit(cache=True)
def evaluate(n, alloc, fam, reuse, cF, cB0, redc, prm, res, perT, fl0, flr):
    """Full iteration. res[:] <- T, comp, comm, E_laser, E_tune, E_dyn, E_leak, slotsF, slotsB,
    maxroute, maxfan, wireF, wireB, commF, commB.  perT[t] <- duration of period t (2L)."""
    L = alloc.shape[0]
    m = int(prm[0]); bwdT = prm[15] > 0.5
    st = starts_of(alloc, m, fam, reuse)
    out = np.zeros(9)
    comp = 0.0; commF = 0.0; commB = 0.0; Elas = 0.0; Etun = 0.0; Edyn = 0.0
    sF = 0.0; sB = 0.0; mr = 0.0; mf = 0.0; wF = 0.0; wB = 0.0
    for t in range(2 * L):
        perT[t] = 0.0
    for i in range(L):
        a = alloc[i]
        cf = cF[i, a]
        cb = cB0[i, a]
        if i < L - 1:
            cb += redc[i, a] * (alloc[i + 1] - 1)
        comp += cf + cb
        perT[i] += cf
        perT[2 * L - 1 - i] += cb
    for i in range(L - 1):          # F_{i+1} -> F_{i+2}  (0-based layer i -> i+1)
        transition(m, st[i], alloc[i], st[i + 1], alloc[i + 1], 0, n[i + 1], n[i + 2], prm, out)
        commF += out[0]; Elas += out[2]; Etun += out[3]; Edyn += out[4]; sF += out[1]
        wF += out[7]
        if out[5] > mr: mr = out[5]
        if out[6] > mf: mf = out[6]
        perT[i] += out[0]
    for i in range(1, L):           # B_{i+1} -> B_i  (0-based layer i -> i-1)
        if bwdT:
            transition(m, st[i], alloc[i], st[i - 1], alloc[i - 1], 0, n[i + 1], n[i], prm, out)
        else:
            transition(m, st[i], alloc[i], st[i - 1], alloc[i - 1], 1, n[i + 1], n[i], prm, out)
        commB += out[0]; Elas += out[2]; Etun += out[3]; Edyn += out[4]; sB += out[1]
        wB += out[7]
        if out[5] > mr: mr = out[5]
        if out[6] > mf: mf = out[6]
        perT[2 * L - 1 - i] += out[0]
    T = prm[14] + comp + commF + commB
    res[0] = T; res[1] = comp; res[2] = commF + commB; res[3] = Elas; res[4] = Etun; res[5] = Edyn
    res[6] = prm[13] * m * T; res[7] = sF; res[8] = sB; res[9] = mr; res[10] = mf
    res[11] = wF; res[12] = wB; res[13] = commF; res[14] = commB
    fl = 0.0
    for i in range(L):
        fl += fl0[i]
        if i < L - 1:
            fl += flr[i] * (alloc[i + 1] - 1)
    res[15] = fl * prm[16]
    res[16] = Elas + Etun + Edyn + res[6] + res[15]          # total energy
    return T


@njit(cache=True)
def obj_only(n, alloc, fam, reuse, cF, cB0, redc, prm, fl0, flr, obj):
    res = np.zeros(17); perT = np.zeros(2 * alloc.shape[0])
    T = evaluate(n, alloc, fam, reuse, cF, cB0, redc, prm, res, perT, fl0, flr)
    if obj == 1:
        return res[16]
    return T


@njit(cache=True)
def coord_search(n, init, fam, reuse, cF, cB0, redc, prm, maxdom, max_sweeps, fl0, flr, obj):
    """Exact one-dimensional minimisation per layer, strict improvement, smaller-count ties."""
    L = init.shape[0]
    alloc = init.copy()
    best = obj_only(n, alloc, fam, reuse, cF, cB0, redc, prm, fl0, flr, obj)
    sweeps = 0
    evals = 1
    for s in range(max_sweeps):
        sweeps += 1
        improved = False
        for i in range(L):
            bi = best; bu = alloc[i]
            cur = alloc[i]
            for u in range(1, maxdom[i] + 1):
                if u == cur:
                    continue
                alloc[i] = u
                T = obj_only(n, alloc, fam, reuse, cF, cB0, redc, prm, fl0, flr, obj)
                evals += 1
                if T < bi - 1e-15 or (abs(T - bi) <= 1e-15 and u < bu):
                    bi = T; bu = u
            if bu != cur and bi < best - 1e-15:
                alloc[i] = bu; best = bi; improved = True
            else:
                alloc[i] = cur
        if not improved:
            break
    return alloc, best, sweeps, evals


@njit(cache=True, parallel=True)
def pair_matrix(li, n, fam, cF, cB0, redc, prm, A, B):
    """t_i(a,b): F transition layer li->li+1, B transition li+1->li, and the R reduction term,
    for all a in 1..A, b in 1..B (0-based layer index li). FM: both arcs start at 0.
    RRM: arc li starts at 0, arc li+1 starts at a (translation invariance)."""
    m = int(prm[0]); bwdT = prm[15] > 0.5
    P = np.full((A + 1, B + 1), np.inf)
    for a in prange(1, A + 1):
        out = np.zeros(9)
        for b in range(1, B + 1):
            s1 = 0 if fam == 0 else a % m
            transition(m, 0, a, s1, b, 0, n[li + 1], n[li + 2], prm, out)
            t = out[0]
            if bwdT:
                transition(m, s1, b, 0, a, 0, n[li + 2], n[li + 1], prm, out)
            else:
                transition(m, s1, b, 0, a, 1, n[li + 2], n[li + 1], prm, out)
            t += out[0]
            t += redc[li, a] * (b - 1)
            P[a, b] = t
    return P


def chain_dp(n, fam: str, cF, cB0, redc, prm, maxdom):
    """Exact minimiser of the iteration time for FM and RRM (Proposition 2)."""
    assert fam in ("FM", "RRM")
    L = len(maxdom)
    na = np.asarray(n, dtype=np.int64)
    f = FAM[fam]
    V = np.full(maxdom[0] + 1, np.inf)
    V[1:] = cF[0, 1:maxdom[0] + 1] + cB0[0, 1:maxdom[0] + 1]
    back = []
    for li in range(L - 1):
        A, B = maxdom[li], maxdom[li + 1]
        P = pair_matrix(li, na, f, cF, cB0, redc, prm, A, B)
        tot = V[:, None] + P                      # (A+1, B+1)
        arg = np.argmin(tot[1:, :], axis=0) + 1   # smallest a on ties
        Vn = tot[arg, np.arange(B + 1)]
        Vn[1:] += cF[li + 1, 1:B + 1] + cB0[li + 1, 1:B + 1]
        Vn[0] = np.inf
        back.append(arg); V = Vn
    last = int(np.argmin(V[1:]) + 1)
    alloc = [last]
    for arg in reversed(back):
        alloc.append(int(arg[alloc[-1]]))
    alloc.reverse()
    return alloc, float(V[last] + prm[14])


# --------------------------------------------------------------------------
# Relaxed initialiser (Eq. relax)
# --------------------------------------------------------------------------
def relaxed_init(n, cfg: Cfg, A_override=None) -> List[float]:
    """m_hat_i = sqrt(A_i/B_i). A_override[i-1], if given, is the single-core time of layer i
    (FP + BP), which is how A_i is calibrated from profiler traces."""
    L = len(n) - 1
    prm = cfg.pack()
    Dfix = prm[7]
    out = []
    for i in range(1, L + 1):
        fF, fB, red = flops_per_neuron(n, i, cfg)
        Ai = n[i] * (fF / cfg.C_F + fB / cfg.C_B) if A_override is None else A_override[i - 1]
        Bi = 0.0
        if i < L:
            Bi += Dfix / cfg.lam
        if i > 1:
            Bi += Dfix / cfg.lam
            if cfg.bwd == "R":
                Bi += 8 * cfg.mu * cfg.psi * n[i - 1] / (cfg.B_lam * cfg.lam)
        if cfg.bwd == "R" and i < L:
            Bi += 0.0  # reduction accumulation couples layers; omitted from the initialiser
        out.append(math.sqrt(Ai / Bi) if Bi > 0 else float(min(n[i], cfg.m)))
    return out


def clip(x, n, m):
    return [int(min(max(1, round(v)), n[i + 1], m)) for i, v in enumerate(x)]


# --------------------------------------------------------------------------
# Schedule metrics and SRAM liveness
# --------------------------------------------------------------------------
def core_sets(alloc, m, fam, reuse=-1):
    st = starts_of(np.asarray(alloc, dtype=np.int64), m, FAM[fam], reuse)
    return [[(int(st[i]) + c) % m for c in range(alloc[i])] for i in range(len(alloc))]


def schedule_metrics(S, m, perT=None):
    L = len(S)
    per = [S[i] for i in range(L)] + [S[i] for i in reversed(range(L))]
    a = np.zeros((2 * L + 2, m), dtype=np.int8)
    for t, st in enumerate(per, start=1):
        a[t, st] = 1
    Z = int(np.abs(np.diff(a.astype(int), axis=0)).sum())
    # longest run of consecutive active periods per core
    run = np.zeros(m, dtype=int); best = np.zeros(m, dtype=int)
    for t in range(1, 2 * L + 1):
        run = np.where(a[t] == 1, run + 1, 0)
        best = np.maximum(best, run)
    out = {"Z": Z, "Rrun": int(best.max())}
    if perT is not None:
        H = (a[1:-1].T.astype(float) * np.asarray(perT)).sum(axis=1)
        out["Hmax"] = float(H.max()); out["Hmean_active"] = float(H[H > 0].mean())
        out["cores_used"] = int((H > 0).sum())
    # local-delivery fraction of forward activation bytes (logical deliveries consumed locally)
    return out


def local_fraction(n, S):
    """rho^{loc,F}: fraction of logical source-to-destination activation bytes delivered locally."""
    L = len(S)
    num = den = 0.0
    for i in range(L - 1):
        a = len(S[i]); base, rem = divmod(n[i + 1], a)
        q = {k: base + (1 if p < rem else 0) for p, k in enumerate(S[i])}
        dst = set(S[i + 1])
        tot = sum(q.values())
        num += sum(q[k] for k in S[i] if k in dst)
        den += len(dst) * tot
    return num / den if den else 0.0


def sram_peak(n, S, cfg: Cfg):
    """Peak bytes on any core by tensor liveness (Table liveness)."""
    L = len(n) - 1; mu, psi = cfg.mu, cfg.psi; T = cfg.bwd == "T"
    m = cfg.m
    per = [("F", i) for i in range(1, L + 1)] + [("B", i) for i in range(L, 0, -1)]
    pos = {p: t for t, p in enumerate(per)}
    Q = np.zeros((L + 1, m))
    for i in range(1, L + 1):
        a = len(S[i - 1]); base, rem = divmod(n[i], a)
        for p, k in enumerate(S[i - 1]):
            Q[i, k] = base + (1 if p < rem else 0)
    best = 0.0; per_max = []
    for t, (ph, li) in enumerate(per):
        tot = np.zeros(m)
        for i in range(1, L + 1):
            q = Q[i]
            own = q > 0
            tot += q * n[i - 1] + q                                   # W_i columns, b_i
            if T and i < L:
                tot += q * n[i + 1]                                   # W_{i+1} rows (replica)
            if pos[("F", i)] <= t <= pos[("B", i)]:
                tot += mu * q + own * mu * n[i - 1]                   # saved A_i rows, received A_{i-1}
            if (ph, li) == ("B", i):
                tot += mu * q + q * n[i - 1] + q                      # Delta_i rows, grad W_i, grad b_i
                if (not T) and i > 1:
                    tot += own * mu * n[i - 1]                        # partial error E_{i-1}^{(k)}
                if T and i < L:
                    tot += own * mu * n[i + 1]                        # received Delta_{i+1}
        pm = float(tot.max()) * psi
        per_max.append(pm); best = max(best, pm)
    return best, per_max


# --------------------------------------------------------------------------
# Analytical electrical 2D mesh (optimistic lower bound)
# --------------------------------------------------------------------------
@njit(cache=True)
def mesh_transition(X, Y, s0, a, d0, b, mode, n_src, n_dst, mu, psi, tree, out):
    """Bit-count bounds for one transition on an X x Y mesh with XY routing; blocks are
    row-major contiguous ranges (no wrap). out <- bits_bound, Hmax.
    bits_bound = max(injection, ejection, vertical-cut/Y, horizontal-cut/X)."""
    halfx = X // 2; halfy = Y // 2
    bs = n_src // a; rs = n_src % a
    bd = n_dst // b; rd = n_dst % b
    # destination statistics
    nd_right = 0; nd_left = 0; nd_bot = 0; nd_top = 0
    q_right = 0.0; q_left = 0.0; q_bot = 0.0; q_top = 0.0
    colsB = np.zeros(X, dtype=np.int64); colsT = np.zeros(X, dtype=np.int64)
    dmin_s = 1 << 30; dmax_s = -(1 << 30); dmin_d = 1 << 30; dmax_d = -(1 << 30)
    qmax_d = 0
    for p in range(b):
        h = d0 + p
        x = h % X; y = h // X
        qh = bd + (1 if p < rd else 0)
        if qh > qmax_d:
            qmax_d = qh
        if x >= halfx:
            nd_right += 1; q_right += qh
        else:
            nd_left += 1; q_left += qh
        if y >= halfy:
            nd_bot += 1; q_bot += qh; colsB[x] = 1
        else:
            nd_top += 1; q_top += qh; colsT[x] = 1
        if x + y < dmin_s: dmin_s = x + y
        if x + y > dmax_s: dmax_s = x + y
        if x - y < dmin_d: dmin_d = x - y
        if x - y > dmax_d: dmax_d = x - y
    ncolB = 0; ncolT = 0
    for x in range(X):
        ncolB += colsB[x]; ncolT += colsT[x]
    inj = 0.0; tot_src = 0.0
    LR = 0.0; RL = 0.0; TB = 0.0; BT = 0.0
    smin_s = 1 << 30; smax_s = -(1 << 30); smin_d = 1 << 30; smax_d = -(1 << 30)
    nsrc_left = 0; nsrc_top = 0
    for p in range(a):
        k = s0 + p
        x = k % X; y = k // X
        indst = 1 if (k >= d0 and k < d0 + b) else 0
        fan = b - indst
        if mode == 0:
            byt = mu * psi * (bs + (1 if p < rs else 0))
        else:
            own = (bd + (1 if (k - d0) < rd else 0)) if indst == 1 else 0
            byt = mu * psi * (n_dst - own)
        if fan == 0:
            byt = 0.0
        if x + y < smin_s: smin_s = x + y
        if x + y > smax_s: smax_s = x + y
        if x - y < smin_d: smin_d = x - y
        if x - y > smax_d: smax_d = x - y
        if x < halfx: nsrc_left += 1
        if y < halfy: nsrc_top += 1
        if mode == 0:
            tot_src += byt
            if tree:
                inj = max(inj, byt)
            else:
                inj = max(inj, byt * fan)
            if x < halfx:
                if tree:
                    LR += byt if nd_right > 0 else 0.0
                else:
                    LR += byt * nd_right
            else:
                if tree:
                    RL += byt if nd_left > 0 else 0.0
                else:
                    RL += byt * nd_left
            if y < halfy:
                TB += byt * (ncolB if tree else nd_bot)
            else:
                BT += byt * (ncolT if tree else nd_top)
        else:
            inj = max(inj, byt)
    # ejection: exact per destination
    ej = 0.0
    for p in range(b):
        h = d0 + p
        insrc = (h >= s0 and h < s0 + a)
        if mode == 0:
            own = 0.0
            if insrc and b > 1:
                ps = h - s0
                own = mu * psi * (bs + (1 if ps < rs else 0))
            e = tot_src - own
        else:
            qh = bd + (1 if p < rd else 0)
            e = mu * psi * qh * (a - (1 if insrc else 0))
        if e > ej:
            ej = e
    if mode == 1:
        nsrc_right = a - nsrc_left; nsrc_bot = a - nsrc_top
        LR = nsrc_left * mu * psi * q_right; RL = nsrc_right * mu * psi * q_left
        TB = nsrc_top * mu * psi * q_bot; BT = nsrc_bot * mu * psi * q_top
    vcut = max(LR, RL) / Y
    hcut = max(TB, BT) / X
    bb = max(max(inj, ej), max(vcut, hcut)) * 8.0
    H = max(max(smax_s - dmin_s, dmax_s - smin_s), max(smax_d - dmin_d, dmax_d - smin_d))
    out[0] = bb; out[1] = H


@njit(cache=True)
def mesh_eval(n, alloc, cF, cB0, redc, X, Y, mu, psi, bwdT, tree, D_init, out2):
    """Returns (K, W): T_mesh(B_e) = K + W / B_e, K = compute + D_init + hop latency."""
    L = alloc.shape[0]
    out = np.zeros(2)
    comp = 0.0
    for i in range(L):
        a = alloc[i]
        cb = cB0[i, a]
        if i < L - 1:
            cb += redc[i, a] * (alloc[i + 1] - 1)
        comp += cF[i, a] + cb
    W = 0.0; Hs = 0.0
    for i in range(L - 1):
        mesh_transition(X, Y, 0, alloc[i], 0, alloc[i + 1], 0, n[i + 1], n[i + 2], mu, psi, tree, out)
        W += out[0]; Hs += out[1]
    for i in range(1, L):
        mode = 0 if bwdT else 1
        mesh_transition(X, Y, 0, alloc[i], 0, alloc[i - 1], mode, n[i + 1], n[i], mu, psi, tree, out)
        W += out[0]; Hs += out[1]
    out2[0] = comp + D_init; out2[1] = W; out2[2] = Hs


def mesh_time(n, alloc, cF, cB0, redc, cfg: Cfg, X=25, Y=40, B_e=128 * 3.4e9, t_r_cyc=2, tree=True):
    o = np.zeros(3)
    mesh_eval(np.asarray(n, dtype=np.int64), np.asarray(alloc, dtype=np.int64), cF, cB0, redc,
              X, Y, float(cfg.mu), float(cfg.psi), cfg.bwd == "T", tree, d_init(n, cfg), o)
    K = o[0] + o[2] * t_r_cyc / cfg.f_clk
    return K + o[1] / B_e, K, o[1]


# --------------------------------------------------------------------------
# Convenience layer
# --------------------------------------------------------------------------
def d_init(n, cfg: Cfg) -> float:
    return cfg.mu * n[0] * cfg.psi / cfg.mem_bw


class Model:
    def __init__(self, n, cfg: Cfg, trace=None, trace_net=None):
        """trace: callable (see compute_tables) or None; trace_net: network name whose profiler
        traces (traces/compute_traces.csv) drive the computation times."""
        self.n = list(n); self.cfg = cfg
        self.na = np.asarray(n, dtype=np.int64)
        self.L = len(n) - 1
        if trace_net is not None:
            self.cF, self.cB0, self.redc = compute_tables_from_traces(trace_net, n, cfg)
        else:
            self.cF, self.cB0, self.redc = compute_tables(n, cfg, trace)
        self.prm = cfg.pack(); self.prm[14] = d_init(n, cfg)
        self.maxdom = np.array([min(n[i + 1], cfg.m) for i in range(self.L)], dtype=np.int64)
        self.fl0 = np.zeros(self.L); self.flr = np.zeros(self.L)
        for i in range(1, self.L + 1):
            fF, fB, red = flops_per_neuron(n, i, cfg)
            self.fl0[i - 1] = n[i] * (fF + fB); self.flr[i - 1] = n[i] * red

    def T(self, alloc, fam="FM", reuse=-1, obj=0):
        return obj_only(self.na, np.asarray(alloc, dtype=np.int64), FAM[fam], reuse,
                        self.cF, self.cB0, self.redc, self.prm, self.fl0, self.flr, obj)

    def E(self, alloc, fam="FM", reuse=-1):
        return self.T(alloc, fam, reuse, 1)

    def full(self, alloc, fam="FM", reuse=-1):
        res = np.zeros(17); perT = np.zeros(2 * self.L)
        a = np.asarray(alloc, dtype=np.int64)
        evaluate(self.na, a, FAM[fam], reuse, self.cF, self.cB0, self.redc, self.prm, res, perT,
                 self.fl0, self.flr)
        keys = ["T", "comp", "comm", "E_laser", "E_tune", "E_dyn", "E_leak", "slotsF", "slotsB",
                "maxroute", "maxfan", "wireF", "wireB", "commF", "commB"]
        d = dict(zip(keys, res.tolist()))
        d["E_net"] = d["E_laser"] + d["E_tune"] + d["E_dyn"] + d["E_leak"]
        d["flops"] = total_flops(self.n, list(alloc), self.cfg)
        d["E_comp"] = d["flops"] * self.cfg.E_flop
        assert abs(d["E_comp"] - res[15]) <= 1e-9 * max(1.0, d["E_comp"])
        d["E_total"] = d["E_net"] + d["E_comp"]
        d["D_init"] = float(self.prm[14])
        d["perT"] = perT.tolist()
        return d

    def coord(self, init, fam="FM", reuse=-1, max_sweeps=100, obj=0):
        a, T, sw, ev = coord_search(self.na, np.asarray(init, dtype=np.int64), FAM[fam], reuse,
                                    self.cF, self.cB0, self.redc, self.prm, self.maxdom, max_sweeps,
                                    self.fl0, self.flr, obj)
        return [int(x) for x in a], float(T), int(sw), int(ev)

    def dp(self, fam="FM"):
        return chain_dp(self.n, fam, self.cF, self.cB0, self.redc, self.prm, list(self.maxdom))

    def _A(self):
        # single-core time of each layer (q = n_i): the calibrated A_i of Eq. (relax)
        return [self.cF[i, 1] + self.cB0[i, 1] for i in range(self.L)]

    def relaxed(self):
        return clip(relaxed_init(self.n, self.cfg, self._A()), self.n, self.cfg.m)

    def relaxed_real(self):
        return relaxed_init(self.n, self.cfg, self._A())

    def fixed(self, u):
        return [int(min(self.n[i + 1], u)) for i in range(self.L)]

    def best_fixed(self, fam="FM", obj=0):
        best = None
        for u in range(1, self.cfg.m + 1):
            T = self.T(self.fixed(u), fam, -1, obj)
            if best is None or T < best[1] - 1e-15:
                best = (u, T)
        return best

    def orrm_best(self, init):
        """ORRM: E[r] rule plus a finite set of fixed reuse settings; best by coordinate search."""
        cands = [-1, 0, 8, 32, 64, 128, 256]
        best = None
        for r in cands:
            a, T, sw, ev = self.coord(init, "ORRM", r)
            if best is None or T < best[1] - 1e-15:
                best = (a, T, r)
        return best
