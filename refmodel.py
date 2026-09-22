"""
Reference implementation of the corrected FCNN-on-ring-ONoC cost model.

Purpose: an executable specification of the equations printed in the revised
manuscript, so that the revived simulator can be checked term by term.
It is NOT the evaluation used for the paper's figures.

Conventions
-----------
* Layers are 1..L (L = ell). n[0] is the input width, n[i] the width of layer i.
* Cores are 0..m-1 on a logical ring. A core set S_i is an ordered list (block order).
* Neuron j (0-based) of layer i is owned by S_i[j mod m_i]  (Algorithm 1).
* Execution order: F_1..F_L, B_L..B_1  (periods t = 1..2L).
* Times are in seconds unless a name ends with _cyc.
* Backward collective:
    'R' = partial-error reduction (one copy of W_i, stored by columns);
    'T' = replicated-transpose broadcast (W_{i+1} also stored by rows on layer-i owners).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple
import numpy as np


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    m: int = 1000                 # cores on the ring
    lam: int = 64                 # wavelengths lambda_max
    mu: int = 8                   # mini-batch size
    psi: int = 8                  # bytes per scalar (double)
    f_clk: float = 3.4e9          # core clock (Hz)
    C_F: float = 6e9              # sustained FP throughput per core (FLOP/s)
    C_B: float = 6e9              # sustained BP throughput per core (FLOP/s)
    B_lam: float = 40e9           # bit rate per wavelength (bit/s)
    flit: int = 16                # flit size (bytes)
    D_cfg: float = 10e-9          # MRR (re)configuration latency per slot (s)  [placeholder]
    D_eo_cyc: int = 1             # E/O conversion (cycles)
    D_oe_cyc: int = 1             # O/E conversion (cycles)
    t_hop_cyc: float = 0.0        # propagation per ring hop (cycles); 0 => ignore
    ser_cyc_override: int | None = None  # force cycles/flit (e.g. 2 for the old abstract value)
    D_init: float = 0.0           # period-0 load time (constant w.r.t. allocation)
    bwd: str = "R"                # 'R' or 'T'
    include_lower_order: bool = True

    @property
    def T_flit_cyc(self) -> int:
        if self.ser_cyc_override is not None:
            return self.ser_cyc_override
        return math.ceil(8 * self.flit * self.f_clk / self.B_lam)

    def D_ser(self, nbytes: float) -> float:
        """Serialisation time of one source payload, flit-granular."""
        return math.ceil(nbytes / self.flit) * self.T_flit_cyc / self.f_clk

    @property
    def D_fix(self) -> float:
        """Per-slot terms independent of the allocation (propagation excluded)."""
        return self.D_cfg + (self.D_eo_cyc + self.D_oe_cyc) / self.f_clk


# --------------------------------------------------------------------------
# Mapping families
# --------------------------------------------------------------------------
def blocks(alloc: Sequence[int], m: int, family: str, reuse: int | None = None) -> List[List[int]]:
    """Return S_1..S_L as ordered core lists (block order) for a mapping family.

    ORRM follows Eqs. (ol), (shift), (id) of the manuscript; if `reuse` is given it
    replaces round(E[r]) (finite reuse-parameter enumeration)."""
    L = len(alloc)
    starts = [0] * L
    if family == "FM":
        starts = [0] * L
    elif family == "RRM":
        for i in range(1, L):
            starts[i] = (starts[i - 1] + alloc[i - 1]) % m
    elif family == "ORRM":
        tot = sum(alloc)
        Er = 0.0 if tot <= m or L == 1 else (tot - m) / (L - 1)
        target = int(round(Er)) if reuse is None else reuse
        r_prev = 0
        for i in range(1, L):
            r = min(target, alloc[i - 1] - r_prev, alloc[i])
            r = max(r, 0)
            starts[i] = (starts[i - 1] + alloc[i - 1] - r) % m
            r_prev = r
    else:
        raise ValueError(family)
    return [[(starts[i] + c) % m for c in range(alloc[i])] for i in range(L)]


def loads(n_i: int, S: Sequence[int]) -> Dict[int, int]:
    """q_{i,k}: neurons of the layer owned by each core under round-robin assignment."""
    a = len(S)
    base, rem = divmod(n_i, a)
    return {k: base + (1 if pos < rem else 0) for pos, k in enumerate(S)}


def d_ring(u: int, v: int, m: int) -> int:
    x = (v - u) % m
    return min(x, m - x)


# --------------------------------------------------------------------------
# Communication: one transition (source set -> destination set)
# --------------------------------------------------------------------------
@dataclass
class Transition:
    slots: int
    time: float
    emitters: int
    bytes_on_wire: float           # sum of source payloads actually emitted
    logical_bytes: float           # sum over (source, destination) pairs incl. local
    local_bytes: float             # part of logical_bytes consumed on the same core
    max_route: int
    per_slot: List[Tuple[float, int, float]] = field(default_factory=list)  # (max payload, route, slot time)


def transition(src: Sequence[int], dst: Sequence[int], payload: Dict[int, float],
               P: Params, local_payload: Dict[int, float] | None = None) -> Transition:
    """Generic slotted multicast: every source k with a non-empty remote destination set
    V_k = dst \\ {k} emits payload[k] bytes on one wavelength; at most lam sources per slot;
    sources are grouped in block order; a slot lasts D_cfg + D_EO + D_OE + D_prop(route) +
    D_ser(max payload in slot)."""
    m = P.m
    dset = set(dst)
    emit = [k for k in src if (dset - {k})]
    per_slot = []
    total = 0.0
    maxroute = 0
    for g0 in range(0, len(emit), P.lam):
        G = emit[g0:g0 + P.lam]
        bmax = max(payload[k] for k in G)
        route = max(max(d_ring(k, v, m) for v in dset if v != k) for k in G)
        t = P.D_fix + P.t_hop_cyc * route / P.f_clk + P.D_ser(bmax)
        per_slot.append((bmax, route, t))
        total += t
        maxroute = max(maxroute, route)
    wire = sum(payload[k] for k in emit)
    lp = local_payload if local_payload is not None else payload
    logical = sum(lp[k] * len(dset) for k in src)
    local = sum(lp[k] for k in src if k in dset)
    return Transition(len(per_slot), total, len(emit), wire, logical, local, maxroute, per_slot)


# --------------------------------------------------------------------------
# Full iteration
# --------------------------------------------------------------------------
@dataclass
class StepResult:
    T_step: float
    comp_F: List[float]
    comp_B: List[float]
    gF: List[Transition]     # gF[i-1] : F_i -> F_{i+1}, i = 1..L-1
    gB: List[Transition]     # gB[i-2] : B_i -> B_{i-1}, i = 2..L
    S: List[List[int]]

    @property
    def comm(self) -> float:
        return sum(t.time for t in self.gF) + sum(t.time for t in self.gB)

    @property
    def comp(self) -> float:
        return sum(self.comp_F) + sum(self.comp_B)


def per_neuron_flops(n: Sequence[int], i: int, alloc: Sequence[int], P: Params) -> Tuple[float, float]:
    """Leading-order (+ optional lower-order) FLOPs per owned neuron of layer i (1-based)."""
    L = len(n) - 1
    mu = P.mu
    aF = 2 * mu * n[i - 1]
    hF = 2 * mu if P.include_lower_order else 0          # bias add + activation
    if P.bwd == "R":
        aB = 2 * mu * n[i - 1] + (2 * mu * n[i - 1] if i > 1 else 0)
        red = mu * (alloc[i + 1] - 1) if i < L else 0      # accumulate |S_{i+1}| partial errors
    else:  # 'T'
        aB = 2 * mu * n[i - 1] + (4 * mu * n[i + 1] if i < L else 0)
        red = 0
    hB = (3 * mu + 2 * (n[i - 1] + 1)) if P.include_lower_order else 0  # deriv, bias grad, SGD update
    if P.bwd == "T" and P.include_lower_order and i < L:
        hB += 2 * n[i + 1]                                  # update the replicated rows
    return aF + hF, aB + red + hB


def evaluate(n: Sequence[int], alloc: Sequence[int], P: Params, family: str,
             reuse: int | None = None) -> StepResult:
    L = len(n) - 1
    assert len(alloc) == L
    S = blocks(alloc, P.m, family, reuse)
    q = [None] + [loads(n[i], S[i - 1]) for i in range(1, L + 1)]
    compF, compB = [], []
    for i in range(1, L + 1):
        fF, fB = per_neuron_flops(n, i, [None] + list(alloc), P)
        qmax = max(q[i].values())
        compF.append(fF * qmax / P.C_F)
        compB.append(fB * qmax / P.C_B)
    gF, gB = [], []
    for i in range(1, L):                   # activation broadcast A_i : S_i -> S_{i+1}
        pay = {k: P.mu * P.psi * q[i][k] for k in S[i - 1]}
        gF.append(transition(S[i - 1], S[i], pay, P))
    for i in range(2, L + 1):               # backward transition B_i -> B_{i-1}
        if P.bwd == "R":
            Sprev = set(S[i - 2])
            pay = {k: P.mu * P.psi * (n[i - 1] - (q[i - 1][k] if k in Sprev else 0)) for k in S[i - 1]}
        else:
            pay = {k: P.mu * P.psi * q[i][k] for k in S[i - 1]}
        gB.append(transition(S[i - 1], S[i - 2], pay, P))
    T = P.D_init + sum(compF) + sum(compB) + sum(t.time for t in gF) + sum(t.time for t in gB)
    return StepResult(T, compF, compB, gF, gB, S)


# --------------------------------------------------------------------------
# Schedule-derived mapping metrics
# --------------------------------------------------------------------------
def period_sets(S: List[List[int]]) -> List[set]:
    L = len(S)
    return [set(S[i]) for i in range(L)] + [set(S[i]) for i in reversed(range(L))]


def metrics(S: List[List[int]], m: int, durations: Sequence[float] | None = None) -> Dict[str, float]:
    per = period_sets(S)
    a = np.zeros((len(per) + 2, m), dtype=int)
    for t, st in enumerate(per, start=1):
        a[t, list(st)] = 1
    Z = int(np.abs(np.diff(a, axis=0)).sum())
    runs = 0
    for k in range(m):
        cur = best = 0
        for t in range(1, len(per) + 1):
            cur = cur + 1 if a[t, k] else 0
            best = max(best, cur)
        runs = max(runs, best)
    L = len(S)
    Lmax = 0
    for i in range(L - 1):
        for u in S[i]:
            for v in S[i + 1]:
                if u != v:
                    Lmax = max(Lmax, d_ring(u, v, m))
    out = {"Z": Z, "R": runs, "Lmax": Lmax}
    if durations is not None:
        H = (a[1:-1].T * np.asarray(durations)).sum(axis=1)
        out["Hmax"] = float(H.max())
    return out


# --------------------------------------------------------------------------
# SRAM by tensor liveness
# --------------------------------------------------------------------------
def sram_peak(n: Sequence[int], S: List[List[int]], P: Params) -> Tuple[float, int, List[float]]:
    """Peak bytes on any core, the core, and the per-period maximum.

    Live tensors on core k (layer i owned by k with q = q_{i,k}):
      persistent : W_i cols q*n_{i-1}, b_i q  [+ 'T': W_{i+1} rows q*n_{i+1}]
      F_i .. B_i : saved own A_i rows mu*q ; received full A_{i-1} mu*n_{i-1}
      B_i only   : Delta_i rows mu*q ; grad W_i q*n_{i-1} ; grad b_i q ;
                   'R': partial E_{i-1} mu*n_{i-1} (i>1) ; 'T': received Delta_{i+1} mu*n_{i+1} (i<L)
    """
    L = len(n) - 1
    mu, psi = P.mu, P.psi
    q = [None] + [loads(n[i], S[i - 1]) for i in range(1, L + 1)]
    cores = set().union(*[set(s) for s in S])
    periods = [("F", i) for i in range(1, L + 1)] + [("B", i) for i in range(L, 0, -1)]
    pos = {p: t for t, p in enumerate(periods)}
    per_period_max = []
    best, best_core = 0.0, -1
    for t, (ph, li) in enumerate(periods):
        pmax = 0.0
        for k in cores:
            tot = 0
            for i in range(1, L + 1):
                qi = q[i].get(k, 0)
                if qi == 0:
                    continue
                tot += qi * n[i - 1] + qi
                if P.bwd == "T" and i < L:
                    tot += qi * n[i + 1]
                if pos[("F", i)] <= t <= pos[("B", i)]:
                    tot += mu * qi + mu * n[i - 1]
                if (ph, li) == ("B", i):
                    tot += mu * qi + qi * n[i - 1] + qi
                    if P.bwd == "R" and i > 1:
                        tot += mu * n[i - 1]
                    if P.bwd == "T" and i < L:
                        tot += mu * n[i + 1]
            b = tot * psi
            if b > pmax:
                pmax = b
            if b > best:
                best, best_core = b, k
        per_period_max.append(pmax)
    return best, best_core, per_period_max


# --------------------------------------------------------------------------
# Relaxed initializer
# --------------------------------------------------------------------------
def relaxed_init(n: Sequence[int], P: Params) -> List[float]:
    """m_hat_i = sqrt(A_i / B_i) (balanced loads, no ceilings, propagation and the
    m_{i+1}-dependent reduction-accumulation term omitted)."""
    L = len(n) - 1
    mu, psi = P.mu, P.psi
    out = []
    for i in range(1, L + 1):
        aF = 2 * mu * n[i - 1]
        if P.bwd == "R":
            aB = 2 * mu * n[i - 1] + (2 * mu * n[i - 1] if i > 1 else 0)
        else:
            aB = 2 * mu * n[i - 1] + (4 * mu * n[i + 1] if i < L else 0)
        A = n[i] * (aF / P.C_F + aB / P.C_B)
        B = 0.0
        if i < L:
            B += P.D_fix / P.lam
        if i > 1:
            B += P.D_fix / P.lam
            if P.bwd == "R":
                B += 8 * mu * psi * n[i - 1] / (P.B_lam * P.lam)
        out.append(math.sqrt(A / B) if B > 0 else float(min(n[i], P.m)))
    return out


def clip_alloc(x: Sequence[float], n: Sequence[int], m: int) -> List[int]:
    return [int(min(max(1, round(v)), n[i + 1], m)) for i, v in enumerate(x)]


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------
def coordinate_search(n, P, family, init, reuse=None, cand=None, max_sweeps=50):
    """Exact 1-D evaluation per layer, strict-improvement acceptance, smaller-count ties."""
    L = len(n) - 1
    alloc = list(init)
    best = evaluate(n, alloc, P, family, reuse).T_step
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        improved = False
        for i in range(L):
            dom = cand[i] if cand is not None else range(1, min(n[i + 1], P.m) + 1)
            bi, bu = best, alloc[i]
            for u in dom:
                if u == alloc[i]:
                    continue
                trial = alloc.copy(); trial[i] = u
                T = evaluate(n, trial, P, family, reuse).T_step
                if T < bi - 1e-15 or (abs(T - bi) <= 1e-15 and u < bu):
                    bi, bu = T, u
            if bu != alloc[i] and bi < best - 1e-15:
                alloc[i], best = bu, bi
                improved = True
        if not improved:
            break
    return alloc, best, sweeps


def chain_dp(n, P, family, cand=None):
    """Exact minimiser for FM and RRM: T = sum_i c_i(m_i) + sum_i t_i(m_i, m_{i+1}).
    Valid because FM blocks are prefixes and RRM blocks are translates, and every cost
    term depends on absolute positions only through ring differences."""
    assert family in ("FM", "RRM")
    L = len(n) - 1
    doms = [list(cand[i]) if cand is not None else list(range(1, min(n[i + 1], P.m) + 1)) for i in range(L)]

    def unary(i, a):   # layer i (1-based): FP compute + BP compute without the reduction term
        q = math.ceil(n[i] / a)
        fF, fB0 = per_neuron_flops(n, i, [None] + [1] * (L + 1), P)   # red term with m_{i+1}=1 -> 0
        return fF * q / P.C_F + fB0 * q / P.C_B

    def pair(i, a, b):  # layers i, i+1: gF_i, gB_{i+1}, and reduction accumulation on layer i
        S = blocks([a, b], P.m, family)
        qa, qb = loads(n[i], S[0]), loads(n[i + 1], S[1])
        payF = {k: P.mu * P.psi * qa[k] for k in S[0]}
        tF = transition(S[0], S[1], payF, P).time
        if P.bwd == "R":
            s0 = set(S[0])
            payB = {k: P.mu * P.psi * (n[i] - (qa[k] if k in s0 else 0)) for k in S[1]}
            red = P.mu * (b - 1) * max(qa.values()) / P.C_B
        else:
            payB = {k: P.mu * P.psi * qb[k] for k in S[1]}
            red = 0.0
        tB = transition(S[1], S[0], payB, P).time
        return tF + tB + red

    V = {a: unary(1, a) for a in doms[0]}
    back = []
    for i in range(1, L):
        newV, arg = {}, {}
        for b in doms[i]:
            u = unary(i + 1, b)
            best, ba = None, None
            for a in doms[i - 1]:
                c = V[a] + pair(i, a, b)
                if best is None or c < best - 1e-15:
                    best, ba = c, a
            newV[b], arg[b] = best + u, ba
        back.append(arg); V = newV
    last = min(V, key=lambda b: (V[b], b))
    alloc = [last]
    for arg in reversed(back):
        alloc.append(arg[alloc[-1]])
    alloc.reverse()
    return alloc, V[last] + P.D_init


# --------------------------------------------------------------------------
# Optimistic analytical electrical mesh baseline (lower bound on ENoC time)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshParams:
    X: int = 25
    Y: int = 40
    B_e: float = 128 * 3.4e9      # link bit rate per direction (bit/s): 16-B flit per cycle
    t_r_cyc: int = 2              # router + link traversal per hop (cycles)
    f_clk: float = 3.4e9


def mesh_xy(k: int, M: MeshParams) -> Tuple[int, int]:
    return k % M.X, k // M.X


def enoc_transition(src, dst, payload: Dict[int, float], M: MeshParams) -> float:
    """Lower bound: ideal tree multicast under XY routing, perfect pipelining.
    T >= max(injection, ejection, bisection) + H_max * t_r."""
    dset = set(dst)
    inj = max((payload[k] for k in src if dset - {k}), default=0.0) * 8 / M.B_e
    ej_bytes = {}
    for k in src:
        for h in dset:
            if h != k:
                ej_bytes[h] = ej_bytes.get(h, 0.0) + payload[k]
    ej = max(ej_bytes.values(), default=0.0) * 8 / M.B_e
    # vertical bisection between columns X/2-1 and X/2 : 2*Y links (one per row per direction)
    half = M.X // 2
    cross = {+1: 0.0, -1: 0.0}
    for k in src:
        xk, _ = mesh_xy(k, M)
        right = any(mesh_xy(h, M)[0] >= half for h in dset if h != k)
        left = any(mesh_xy(h, M)[0] < half for h in dset if h != k)
        if xk < half and right:
            cross[+1] += payload[k]
        if xk >= half and left:
            cross[-1] += payload[k]
    bis = max(cross.values()) * 8 / (M.Y * M.B_e)
    H = 0
    for k in src:
        xk, yk = mesh_xy(k, M)
        for h in dset:
            if h != k:
                xh, yh = mesh_xy(h, M)
                H = max(H, abs(xk - xh) + abs(yk - yh))
    return max(inj, ej, bis) + H * M.t_r_cyc / M.f_clk


def evaluate_enoc(n, alloc, P: Params, M: MeshParams, family="FM") -> float:
    """Same compute and ownership as the ONoC model; only the transitions change.
    Mapping: logical ring index k -> row-major mesh position."""
    L = len(n) - 1
    Pm = replace(P, m=M.X * M.Y)
    S = blocks(alloc, Pm.m, family)
    q = [None] + [loads(n[i], S[i - 1]) for i in range(1, L + 1)]
    T = P.D_init
    for i in range(1, L + 1):
        fF, fB = per_neuron_flops(n, i, [None] + list(alloc), P)
        qmax = max(q[i].values())
        T += fF * qmax / P.C_F + fB * qmax / P.C_B
    for i in range(1, L):
        pay = {k: P.mu * P.psi * q[i][k] for k in S[i - 1]}
        T += enoc_transition(S[i - 1], S[i], pay, M)
    for i in range(2, L + 1):
        if P.bwd == "R":
            # reduce-scatter on a unicast network: each source sends only the destination's rows;
            # bound uses per-destination segments.
            Sprev = S[i - 2]
            src = S[i - 1]
            dset = set(Sprev)
            inj = max(P.mu * P.psi * (n[i - 1] - (q[i - 1][k] if k in dset else 0)) for k in src) * 8 / M.B_e
            ej = max(P.mu * P.psi * q[i - 1][h] * sum(1 for k in src if k != h) for h in Sprev) * 8 / M.B_e
            half = M.X // 2
            right = sum(q[i - 1][h] for h in Sprev if mesh_xy(h, M)[0] >= half)
            left = sum(q[i - 1][h] for h in Sprev if mesh_xy(h, M)[0] < half)
            srcL = sum(1 for k in src if mesh_xy(k, M)[0] < half)
            srcR = len(src) - srcL
            bis = max(srcL * right, srcR * left) * P.mu * P.psi * 8 / (M.Y * M.B_e)
            H = max(abs(mesh_xy(k, M)[0] - mesh_xy(h, M)[0]) + abs(mesh_xy(k, M)[1] - mesh_xy(h, M)[1])
                    for k in src for h in Sprev)
            T += max(inj, ej, bis) + H * M.t_r_cyc / M.f_clk
        else:
            pay = {k: P.mu * P.psi * q[i][k] for k in S[i - 1]}
            T += enoc_transition(S[i - 1], S[i - 2], pay, M)
    return T


NETS = {
    "NN1": [784, 1000, 500, 10],
    "NN2": [784, 1500, 784, 1000, 500, 10],
    "NN3": [784, 2000, 1500, 784, 1000, 500, 10],
    "NN4": [784, 2500, 2000, 1500, 784, 1000, 500, 10],
    "NN5": [1024, 4000, 1000, 4000, 10],
    "NN6": [1024, 4000, 1000, 4000, 1000, 4000, 1000, 4000, 10],
}


# --------------------------------------------------------------------------
# Vectorised evaluation for contiguous-arc mapping families (same semantics)
# --------------------------------------------------------------------------
def _arc_routes(src: np.ndarray, d0: int, b: int, m: int) -> np.ndarray:
    """max_{v in arc(d0,b), v != k} d_ring(k, v) for each source k (0 if no such v)."""
    off = (src - d0) % m                  # position of k relative to the arc start
    last = b - 1
    d_first = np.minimum(off % m, (-off) % m)
    d_last = np.minimum((off - last) % m, (last - off) % m)
    r = np.maximum(d_first, d_last)
    anti = (off + m // 2) % m             # antipode offset(s)
    anti2 = (off + (m + 1) // 2) % m
    has_anti = (anti <= last) | (anti2 <= last)
    r = np.where(has_anti, m // 2, r)
    if b == 1:
        r = np.where(off == 0, 0, r)
    return r


def transition_arc(src_start, a, dst_start, b, payload: np.ndarray, P: Params) -> Tuple[float, int]:
    m = P.m
    src = (src_start + np.arange(a)) % m
    emit = np.ones(a, dtype=bool)
    if b == 1:
        emit = src != dst_start
    pay = payload[emit]
    if pay.size == 0:
        return 0.0, 0
    nslots = -(-pay.size // P.lam)
    pad = nslots * P.lam - pay.size
    bmax = np.pad(pay, (0, pad)).reshape(nslots, P.lam).max(axis=1)
    ser = np.ceil(bmax / P.flit) * P.T_flit_cyc / P.f_clk
    t = nslots * P.D_fix + ser.sum()
    if P.t_hop_cyc:
        rt = _arc_routes(src[emit], dst_start, b, m)
        rmax = np.pad(rt, (0, pad)).reshape(nslots, P.lam).max(axis=1)
        t += (P.t_hop_cyc * rmax / P.f_clk).sum()
    return float(t), int(nslots)


def starts_of(alloc, m, family, reuse=None):
    L = len(alloc)
    st = [0] * L
    if family == "RRM":
        for i in range(1, L):
            st[i] = (st[i - 1] + alloc[i - 1]) % m
    elif family == "ORRM":
        tot = sum(alloc)
        Er = 0.0 if tot <= m or L == 1 else (tot - m) / (L - 1)
        target = int(round(Er)) if reuse is None else reuse
        rp = 0
        for i in range(1, L):
            r = max(0, min(target, alloc[i - 1] - rp, alloc[i]))
            st[i] = (st[i - 1] + alloc[i - 1] - r) % m
            rp = r
    return st


def _q_arr(n_i, a):
    base, rem = divmod(n_i, a)
    return base + (np.arange(a) < rem)


def fast_terms(n, alloc, P: Params, family, reuse=None):
    """Returns (compF, compB, gF_times, gB_times, slotsF, slotsB)."""
    L = len(n) - 1
    m = P.m
    st = starts_of(alloc, m, family, reuse)
    A = [None] + list(alloc)
    compF, compB = [], []
    for i in range(1, L + 1):
        fF, fB = per_neuron_flops(n, i, A + [None], P)
        q = -(-n[i] // alloc[i - 1])
        compF.append(fF * q / P.C_F); compB.append(fB * q / P.C_B)
    gF, gB, sF, sB = [], [], [], []
    for i in range(1, L):
        a, b = alloc[i - 1], alloc[i]
        pay = P.mu * P.psi * _q_arr(n[i], a).astype(float)
        t, s = transition_arc(st[i - 1], a, st[i], b, pay, P)
        gF.append(t); sF.append(s)
    for i in range(2, L + 1):
        a, b = alloc[i - 1], alloc[i - 2]          # sources: layer i ; destinations: layer i-1
        if P.bwd == "R":
            src = (st[i - 1] + np.arange(a)) % m
            posd = (src - st[i - 2]) % m
            inS = posd < b
            qd = _q_arr(n[i - 1], b)
            own = np.where(inS, qd[np.minimum(posd, b - 1)], 0)
            pay = P.mu * P.psi * (n[i - 1] - own).astype(float)
        else:
            pay = P.mu * P.psi * _q_arr(n[i], a).astype(float)
        t, s = transition_arc(st[i - 1], a, st[i - 2], b, pay, P)
        gB.append(t); sB.append(s)
    return compF, compB, gF, gB, sF, sB


def T_fast(n, alloc, P, family, reuse=None):
    cF, cB, gF, gB, _, _ = fast_terms(n, alloc, P, family, reuse)
    return P.D_init + sum(cF) + sum(cB) + sum(gF) + sum(gB)


def coordinate_search_fast(n, P, family, init, reuse=None, max_sweeps=50):
    L = len(n) - 1
    alloc = [int(x) for x in init]
    best = T_fast(n, alloc, P, family, reuse)
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        improved = False
        for i in range(L):
            bi, bu = best, alloc[i]
            for u in range(1, min(n[i + 1], P.m) + 1):
                if u == alloc[i]:
                    continue
                trial = alloc.copy(); trial[i] = u
                T = T_fast(n, trial, P, family, reuse)
                if T < bi - 1e-15 or (abs(T - bi) <= 1e-15 and u < bu):
                    bi, bu = T, u
            if bu != alloc[i] and bi < best - 1e-15:
                alloc[i], best, improved = bu, bi, True
        if not improved:
            break
    return alloc, best, sweeps
