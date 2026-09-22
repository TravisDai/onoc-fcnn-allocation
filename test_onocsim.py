"""Cross-checks: the numba simulator must agree with the independent reference model."""
import itertools, random
import numpy as np
import onocsim as S
import refmodel as R


def _pair(n, m, lam, mu, bwd, D_cfg=10e-9, hop_mm=0.63):
    cfg = S.Cfg(m=m, lam=lam, mu=mu, bwd=bwd, D_cfg=D_cfg, hop_mm=hop_mm)
    P = R.Params(m=m, lam=lam, mu=mu, bwd=bwd, D_cfg=D_cfg, t_hop_cyc=cfg.t_hop * cfg.f_clk,
                 D_init=S.d_init(n, cfg))
    return cfg, P


def test_iteration_time_matches_reference():
    random.seed(3)
    for _ in range(150):
        m = random.randint(3, 48); L = random.randint(1, 5)
        n = [random.randint(1, 70) for _ in range(L + 1)]
        alloc = [random.randint(1, min(n[i + 1], m)) for i in range(L)]
        cfg, P = _pair(n, m, random.choice([1, 2, 3, 8]), random.choice([1, 4, 32]),
                       random.choice("RT"), hop_mm=random.choice([0.63, 50.0]))
        mod = S.Model(n, cfg)
        for fam in ("FM", "RRM", "ORRM"):
            a = mod.T(alloc, fam)
            b = R.evaluate(n, alloc, P, fam).T_step
            assert abs(a - b) <= 1e-12 * max(b, 1e-12), (fam, n, alloc, a, b)


def test_dp_exact_and_coord_upper_bound():
    random.seed(11)
    for _ in range(8):
        n = [random.randint(3, 11) for _ in range(4)]
        cfg = S.Cfg(m=9, lam=random.choice([2, 3]), mu=random.choice([1, 4]), C_F=2e6, C_B=2e6,
                    D_cfg=2e-6, hop_mm=5000.0, bwd=random.choice("RT"))
        mod = S.Model(n, cfg)
        for fam in ("FM", "RRM"):
            a_dp, T_dp = mod.dp(fam)
            best = min(mod.T(list(al), fam) for al in
                       itertools.product(*[range(1, min(n[i + 1], 9) + 1) for i in range(3)]))
            assert abs(T_dp - best) <= 1e-12 * best
            assert abs(mod.T(a_dp, fam) - T_dp) <= 1e-12 * T_dp
            a_cs, T_cs, _, _ = mod.coord([1, 1, 1], fam)
            assert T_cs >= T_dp - 1e-15


def _mesh_brute(X, Y, src, dst, pay_of, seg_of, mode, tree):
    """Direct evaluation of the bound definitions on explicit core lists."""
    xy = lambda k: (k % X, k // X)
    dset = set(dst)
    inj = ej = 0.0
    LR = RL = TB = BT = 0.0
    recv = {h: 0.0 for h in dst}
    for k in src:
        V = [h for h in dst if h != k]
        if not V:
            continue
        xk, yk = xy(k)
        if mode == 0:
            b = pay_of[k]
            inj = max(inj, b if tree else b * len(V))
            for h in V:
                recv[h] += b
            right = [h for h in V if xy(h)[0] >= X // 2]; left = [h for h in V if xy(h)[0] < X // 2]
            bot = [h for h in V if xy(h)[1] >= Y // 2]; top = [h for h in V if xy(h)[1] < Y // 2]
            if xk < X // 2:
                LR += (b if right else 0) if tree else b * len(right)
            else:
                RL += (b if left else 0) if tree else b * len(left)
            if yk < Y // 2:
                TB += b * (len({xy(h)[0] for h in bot}) if tree else len(bot))
            else:
                BT += b * (len({xy(h)[0] for h in top}) if tree else len(top))
        else:
            inj = max(inj, sum(seg_of[h] for h in V))
            for h in V:
                recv[h] += seg_of[h]
                if xk < X // 2 <= xy(h)[0]: LR += seg_of[h]
                if xy(h)[0] < X // 2 <= xk: RL += seg_of[h]
                if yk < Y // 2 <= xy(h)[1]: TB += seg_of[h]
                if xy(h)[1] < Y // 2 <= yk: BT += seg_of[h]
    ej = max(recv.values()) if recv else 0.0
    H = max((abs(xy(u)[0] - xy(v)[0]) + abs(xy(u)[1] - xy(v)[1]) for u in src for v in dst), default=0)
    return 8 * max(inj, ej, max(LR, RL) / Y, max(TB, BT) / X), H


def test_mesh_bound_matches_definition():
    random.seed(5)
    for _ in range(300):
        X, Y = random.choice([(4, 3), (5, 4), (6, 6)])
        m = X * Y
        a = random.randint(1, m); b = random.randint(1, m)
        n_src = random.randint(a, 3 * m); n_dst = random.randint(b, 3 * m)
        mu, psi = 2, 4
        mode = random.choice([0, 1]); tree = random.choice([True, False])
        src = list(range(a)); dst = list(range(b))
        bs, rs = divmod(n_src, a); bd, rd = divmod(n_dst, b)
        qs = {k: bs + (1 if k < rs else 0) for k in src}
        qd = {h: bd + (1 if h < rd else 0) for h in dst}
        pay = {k: mu * psi * qs[k] for k in src}
        seg = {h: mu * psi * qd[h] for h in dst}
        out = np.zeros(2)
        S.mesh_transition(X, Y, 0, a, 0, b, mode, n_src, n_dst, float(mu), float(psi), tree, out)
        bb, H = _mesh_brute(X, Y, src, dst, pay, seg, mode, tree)
        assert abs(out[0] - bb) <= 1e-9 * max(bb, 1), (X, Y, a, b, mode, tree, out[0], bb)
        assert out[1] == H


def test_sram_and_metrics_match_reference():
    random.seed(9)
    for _ in range(60):
        m = random.randint(3, 30); L = random.randint(1, 5)
        n = [random.randint(1, 50) for _ in range(L + 1)]
        alloc = [random.randint(1, min(n[i + 1], m)) for i in range(L)]
        bwd = random.choice("RT"); mu = random.choice([1, 8])
        cfg = S.Cfg(m=m, mu=mu, bwd=bwd)
        P = R.Params(m=m, mu=mu, bwd=bwd)
        for fam in ("FM", "RRM", "ORRM"):
            Sets = S.core_sets(alloc, m, fam)
            assert Sets == R.blocks(alloc, m, fam)
            a, _ = S.sram_peak(n, Sets, cfg)
            b, _, _ = R.sram_peak(n, Sets, P)
            assert abs(a - b) < 1e-9
            ms = S.schedule_metrics(Sets, m); mr = R.metrics(Sets, m)
            assert ms["Z"] == mr["Z"] and ms["Rrun"] == mr["R"]


def _energy_brute(n, alloc, fam, cfg):
    """Eq. (energy) evaluated directly from explicit core lists."""
    Sets = S.core_sets(alloc, cfg.m, fam)
    L = len(alloc); m = cfg.m; mu, psi = cfg.mu, cfg.psi
    q = [None] + [dict((k, n[i] // alloc[i - 1] + (1 if p < n[i] % alloc[i - 1] else 0))
                       for p, k in enumerate(Sets[i - 1])) for i in range(1, L + 1)]
    Tfl = cfg.T_flit_cyc / cfg.f_clk; Dfix = cfg.D_cfg + 2 / cfg.f_clk
    E = 0.0; comm = 0.0
    trans = [(Sets[i], Sets[i + 1], {k: mu * psi * q[i + 1][k] for k in Sets[i]}, True) for i in range(L - 1)]
    for i in range(1, L):
        src, dst = Sets[i], Sets[i - 1]
        if cfg.bwd == "T":
            trans.append((src, dst, {k: mu * psi * q[i + 1][k] for k in src}, True))
        else:
            trans.append((src, dst, {k: mu * psi * (n[i] - q[i].get(k, 0)) for k in src}, False))
    for src, dst, pay, bcast in trans:
        V = {k: [v for v in dst if v != k] for k in src}
        emit = [k for k in src if V[k]]
        for g0 in range(0, len(emit), cfg.lam):
            G = emit[g0:g0 + cfg.lam]
            route = max(max(R.d_ring(k, v, m) for v in V[k]) for k in G)
            tau = Dfix + cfg.t_hop * route + -(-max(pay[k] for k in G) // cfg.flit) * Tfl
            comm += tau
            E += tau * (len(G) * cfg.P_laser_opt / cfg.eta_laser + cfg.P_tune * (len(G) + sum(len(V[k]) for k in G)))
        for k in emit:
            E += 8 * pay[k] * cfg.E_tx + 8 * pay[k] * (len(V[k]) if bcast else 1) * cfg.E_rx
    return E, comm


def test_energy_matches_definition():
    random.seed(21)
    for _ in range(60):
        m = random.randint(3, 30); L = random.randint(1, 4)
        n = [random.randint(1, 40) for _ in range(L + 1)]
        alloc = [random.randint(1, min(n[i + 1], m)) for i in range(L)]
        cfg = S.Cfg(m=m, lam=random.choice([1, 2, 4]), mu=random.choice([1, 8]), bwd=random.choice("RT"))
        mod = S.Model(n, cfg)
        for fam in ("FM", "RRM", "ORRM"):
            d = mod.full(alloc, fam)
            Eb, commb = _energy_brute(n, alloc, fam, cfg)
            assert abs(d["comm"] - commb) <= 1e-12 * max(commb, 1e-12)
            Edyn_tun_las = d["E_laser"] + d["E_tune"] + d["E_dyn"]
            assert abs(Edyn_tun_las - Eb) <= 1e-9 * max(Eb, 1e-15), (Edyn_tun_las, Eb)
            assert abs(d["E_leak"] - cfg.P_leak * m * d["T"]) <= 1e-15
            assert abs(mod.E(alloc, fam) - (d["E_net"] + d["E_comp"])) <= 1e-9 * d["E_net"]


def test_trace_tables():
    """Trace-driven computation: FP uses the profiled kernel; BP is scaled to the modelled FLOPs."""
    n = S.NETS["NN1"]; L = len(n) - 1
    df = S.load_traces()
    for bwd in "RT":
        cfg = S.Cfg(mu=8, bwd=bwd)
        cF, cB0, redc = S.compute_tables_from_traces("NN1", n, cfg, df)
        for i in range(1, L + 1):
            tF = S.per_load_times(df, "NN1", i, 8, n[i])
            tB = S.per_load_times(df, "NN1", 2 * L - i + 1, 8, n[i])
            for a in (1, 7, min(n[i], 1000)):
                q = -(-n[i] // a)
                assert cF[i - 1, a] == tF[q]
                if bwd == "R":
                    ratio = 2 if i > 1 else 1
                else:
                    ratio = 1 + (2 * n[i + 1] / n[i - 1] if i < L else 0)
                assert abs(cB0[i - 1, a] - ratio * tB[q]) <= 1e-15
            vals = [tF[q] for q in sorted(tF)]
            assert all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))   # isotonic
