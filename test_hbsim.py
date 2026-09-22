"""Tests for the Hummingbird-style broadcast network (physhb.py, hbsim.py, hbref.py)."""
import itertools, math, random
from dataclasses import replace
import numpy as np
import physhb as PH
import hbref
from hbsim import HCfg, HModel, htransition


def test_physical_layer():
    p = PH.HBPhys()
    assert PH.f_split(p) == 6                         # 10 log10 6 = 7.8 dB <= 8 dB < 10 log10 7
    assert abs(PH.oma_db(p) - 10 * math.log10((10 ** 0.4 + 1) / (2 * (10 ** 0.4 - 1)))) < 1e-12
    assert abs(PH.sens_dbm(25e9, p) - (-22.3 + 15 * math.log10(2.5))) < 1e-12
    # the fan-out shrinks with the lane rate at a fixed lane power, and is split-limited at high power
    F = [PH.f_max(B, 0.0, "SDM", 1, 2.5, p) for B in (10e9, 25e9, 40e9)]
    assert F[0] >= F[1] >= F[2] >= 1 and F[0] == 6
    assert all(PH.f_max(B, 8.0, "SDM", 1, 2.5, p) == 6 for B in (10e9, 25e9, 40e9))
    # every window that is used closes the budget
    for B in (10e9, 25e9, 40e9):
        for P in (0.0, 8.0):
            for mux, lam in (("SDM", 1), ("WDM", 16)):
                Fm = PH.f_max(B, P, mux, lam, 2.5, p)
                assert PH.il_db(Fm, Fm, mux, lam, 2.5, p) <= PH.budget_db(B, P, p) + 1e-9
                assert 10 * math.log10(Fm) <= p.split_max_db
    assert PH.gamma_wdm_db(16, p) >= p.snr_min_db


def _cfg(rng):
    C = rng.choice([5, 8, 13, 16, 32, 64]); c = rng.choice([1, 2, 4, 16])
    B = rng.choice([10e9, 25e9, 40e9]); P = rng.choice([0.0, 8.0]); mux = rng.choice(["SDM", "WDM"])
    return HCfg(C=C, c=c, k=rng.choice([1, 4, 10, 64]), B=B, P_lane_max_dbm=P, mux=mux,
                pitch_mm=rng.choice([1.25, 2.5, 5.0]), mu=rng.choice([1, 8]),
                ideal=rng.random() < 0.25, B_cl=rng.choice([64 * 3.4e9, 2e9]))


def test_transition_matches_reference():
    rng = random.Random(3)
    out = np.zeros(11)
    for trial in range(500):
        cfg = _cfg(rng); prm = cfg.pack(); C, c = cfg.C, cfg.c
        a = rng.randint(1, C); b = rng.randint(1, C)
        n_src = rng.randint(a * c // 2 + 1, a * c * 3 + 1) if a * c > 1 else rng.randint(1, 9)
        n_dst = rng.randint(b * c // 2 + 1, b * c * 3 + 1) if b * c > 1 else rng.randint(1, 9)
        a = min(a, -(-n_src // c)); b = min(b, -(-n_dst // c))
        s0 = rng.randrange(C); d0 = rng.randrange(C); mode = rng.choice([0, 1])
        htransition(s0, a, d0, b, mode, n_src, n_dst, prm, out)
        ref = hbref.transition_ref(s0, a, d0, b, mode, n_src, n_dst, hbref.params(prm))
        for key, i in (("time", 0), ("E_laser", 1), ("E_tune", 2), ("E_dyn", 3), ("E_fab", 4),
                       ("t_opt", 8), ("t_fab", 9), ("max_load", 10)):
            assert math.isclose(out[i], ref[key], rel_tol=1e-9, abs_tol=1e-18), (trial, key, out[i], ref[key])
        assert int(out[6]) == sum(len(ch["segs"]) for ch in ref["chains"])


def test_chains_cover_every_destination_within_the_fan_out():
    rng = random.Random(5)
    for trial in range(300):
        cfg = replace(_cfg(rng), ideal=False); prm = cfg.pack(); C, c, F = cfg.C, cfg.c, cfg.F
        a = rng.randint(1, C); b = rng.randint(1, C)
        ref = hbref.transition_ref(rng.randrange(C), a, rng.randrange(C), b, rng.choice([0, 1]),
                                   a * c * 2, b * c * 2, hbref.params(prm))
        for ch in ref["chains"]:
            got = set()
            for g, sg in enumerate(ch["segs"]):
                assert sg["end"] - sg["start"] <= F                # a window taps at most F hubs
                if g:
                    assert sg["start"] == ch["segs"][g - 1]["end"]  # the relay starts the next window
                got |= set(sg["receivers"])
            assert got == set(ch["wanted"])


def test_dp_is_exact_small():
    n = [9, 13, 7, 11, 3]
    for bwd in ("R", "T"):
        for fam in ("FM", "RRM"):
            for P in (0.0, 8.0):
                cfg = HCfg(C=7, c=2, k=2, B=40e9, P_lane_max_dbm=P, mu=4, bwd=bwd, pitch_mm=5.0)
                M = HModel("toy", cfg, n=n, analytic=True)
                acl, T = M.dp(fam)
                best = min(M.T(list(x), fam) for x in itertools.product(*[range(1, d + 1) for d in M.maxdom]))
                assert math.isclose(T, best, rel_tol=1e-12)
                assert math.isclose(M.T(acl, fam), T, rel_tol=1e-12)


def test_ideal_never_slower_in_lane_load():
    """Without the fan-out limit no hub relays, so the largest lane load cannot be higher."""
    rng = random.Random(9)
    o1 = np.zeros(11); o2 = np.zeros(11)
    for trial in range(200):
        cfg = replace(_cfg(rng), ideal=False)
        C, c = cfg.C, cfg.c
        a = rng.randint(1, C); b = rng.randint(1, C); s0 = rng.randrange(C); d0 = rng.randrange(C)
        mode = rng.choice([0, 1])
        htransition(s0, a, d0, b, mode, a * c * 2, b * c * 2, cfg.pack(), o1)
        htransition(s0, a, d0, b, mode, a * c * 2, b * c * 2, replace(cfg, ideal=True).pack(), o2)
        assert o2[10] <= o1[10] + 1e-9
