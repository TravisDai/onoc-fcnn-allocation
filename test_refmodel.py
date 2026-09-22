"""Hand-checkable test vectors for the corrected model (run: python -m pytest -q)."""
import itertools, math, random
from dataclasses import replace
import refmodel as R


def test_two_to_four_example():
    # Fig. 2, Scheme 1: F_1 on cores {1,2}, F_2 on cores {1..4}, lambda_max = 2, FM.
    P = R.Params(m=4, lam=2, mu=1)
    S = R.blocks([2, 4], 4, "FM")
    assert S == [[0, 1], [0, 1, 2, 3]]
    pay = {k: 1.0 for k in S[0]}
    t = R.transition(S[0], S[1], pay, P)
    assert t.emitters == 2 and t.slots == 1          # two source broadcasts, not zero
    # old formula: ceil(m_1 (1 - rho_1) / lam) with rho_1 = min(2,4)/2 = 1  -> 0 slots
    rho = min(2, 4) / 2
    assert math.ceil(2 * (1 - rho) / 2) == 0
    # local-delivery fraction: 2 of the 2*4 logical deliveries are local = 1/4 = 1/|S_2|
    assert t.local_bytes / t.logical_bytes == 0.25
    # Scheme 2: 4 -> 4 cores, 4 emitters, 2 slots
    S2 = R.blocks([4, 4], 4, "FM")
    t2 = R.transition(S2[0], S2[1], {k: 1.0 for k in S2[0]}, P)
    assert t2.emitters == 4 and t2.slots == 2


def test_single_destination_degenerate_case():
    P = R.Params(m=8, lam=4)
    t = R.transition([0, 1, 2], [0], {0: 1, 1: 1, 2: 1}, P)
    assert t.emitters == 2                          # core 0 has no remote destination


def test_serialisation_cycles():
    P = R.Params()
    assert P.T_flit_cyc == 11                       # 16 B @ 40 Gb/s @ 3.4 GHz = 10.88 -> 11
    assert R.Params(B_lam=10e9).T_flit_cyc == 44    # 43.52 -> 44


def test_gradient_shapes():
    import numpy as np
    n0, n1, mu = 5, 3, 4
    A0 = np.random.rand(n0, mu); W1 = np.random.rand(n0, n1); D1 = np.random.rand(n1, mu)
    gW = A0 @ D1.T / mu
    assert gW.shape == W1.shape                    # replacement Eq. (3)
    assert (D1 @ A0.T).shape == (n1, n0)           # old G_i = Delta_i X_i^T has the transpose shape


def _orrm_sets(alloc, m):
    return [set(s) for s in R.blocks(alloc, m, "ORRM")]


def test_orrm_three_period_claim_bruteforce():
    """Under the pairwise one-lap condition m_i + m_{i+1} - r_{i+1} <= m, no core lies in three
    consecutive FP periods, although S_{i-1} and S_{i+1} may intersect after wrap-around."""
    found_wrap_overlap = False
    for m in range(3, 11):
        for L in range(3, 6):
            for alloc in itertools.product(range(1, m + 1), repeat=L):
                blocksS = R.blocks(list(alloc), m, "ORRM")
                S = [set(b) for b in blocksS]
                # recover r_i from the construction
                tot = sum(alloc); Er = 0 if tot <= m else (tot - m) / (L - 1)
                r = [0]
                for i in range(1, L):
                    r.append(max(0, min(int(round(Er)), alloc[i - 1] - r[-1], alloc[i])))
                cond = all(alloc[i] + alloc[i + 1] - r[i + 1] <= m for i in range(L - 1))
                if not cond:
                    continue
                for i in range(L - 1):
                    assert len(S[i] & S[i + 1]) == r[i + 1]      # actual overlap = requested
                for i in range(2, L):
                    assert not (S[i - 2] & S[i - 1] & S[i])
                    if S[i - 2] & S[i]:
                        found_wrap_overlap = True
                met = R.metrics(blocksS, m)
                assert met["R"] <= 4
    assert found_wrap_overlap       # Step 1 of the old proof is false


def test_rrm_run_length_bruteforce():
    for m in range(3, 10):
        for L in range(2, 5):
            for alloc in itertools.product(range(1, m + 1), repeat=L):
                if not all(alloc[i] + alloc[i + 1] <= m for i in range(L - 1)):
                    continue
                assert R.metrics(R.blocks(list(alloc), m, "RRM"), m)["R"] <= 2


def test_fm_transitions_closed_form():
    for m in range(3, 9):
        for L in range(2, 5):
            for alloc in itertools.product(range(1, m + 1), repeat=L):
                Z = R.metrics(R.blocks(list(alloc), m, "FM"), m)["Z"]
                assert Z == 2 * (alloc[0] + sum(abs(alloc[i] - alloc[i - 1]) for i in range(1, L)))


def _brute(n, P, fam):
    L = len(n) - 1
    best = None
    for alloc in itertools.product(*[range(1, min(n[i + 1], P.m) + 1) for i in range(L)]):
        T = R.evaluate(n, list(alloc), P, fam).T_step
        if best is None or T < best[1] - 1e-15:
            best = (list(alloc), T)
    return best


def test_chain_dp_is_exact_small():
    random.seed(1)
    for trial in range(6):
        n = [random.randint(4, 12) for _ in range(4)]
        P = R.Params(m=9, lam=random.choice([2, 3]), mu=random.choice([1, 4]), C_F=2e6, C_B=2e6,
                     D_cfg=2e-6, t_hop_cyc=50.0, bwd=random.choice(["R", "T"]))
        for fam in ("FM", "RRM"):
            a_dp, T_dp = R.chain_dp(n, P, fam)
            a_bf, T_bf = _brute(n, P, fam)
            assert abs(T_dp - T_bf) <= 1e-12 * max(1.0, T_bf), (fam, n, a_dp, a_bf, T_dp, T_bf)
            assert abs(R.evaluate(n, a_dp, P, fam).T_step - T_dp) <= 1e-12 * max(1.0, T_dp)
            a_cs, T_cs, _ = R.coordinate_search(n, P, fam, [1] * (len(n) - 1))
            assert T_cs >= T_dp - 1e-15


def test_fast_matches_reference():
    random.seed(7)
    for trial in range(40):
        m = random.randint(5, 40)
        L = random.randint(2, 5)
        n = [random.randint(1, 60) for _ in range(L + 1)]
        alloc = [random.randint(1, min(n[i + 1], m)) for i in range(L)]
        P = R.Params(m=m, lam=random.choice([1, 2, 4, 8]), mu=random.choice([1, 3, 8]),
                     t_hop_cyc=random.choice([0.0, 3.0]), bwd=random.choice(["R", "T"]))
        for fam in ("FM", "RRM", "ORRM"):
            ref = R.evaluate(n, alloc, P, fam).T_step
            fast = R.T_fast(n, alloc, P, fam)
            assert abs(ref - fast) <= 1e-12 * max(ref, 1e-9), (fam, n, alloc, ref, fast)


def test_path_length_closed_forms():
    for m in range(3, 11):
        for L in range(2, 5):
            for alloc in itertools.product(range(1, m + 1), repeat=L):
                al = list(alloc)
                h = m // 2
                fm = R.metrics(R.blocks(al, m, "FM"), m)["Lmax"]
                # pairs with u != v only; a 1->1 transition on the same core has no route
                exp_fm = max((min(max(al[i], al[i + 1]) - 1, h) for i in range(L - 1)), default=0)
                assert fm == exp_fm
                if all(al[i] + al[i + 1] <= m for i in range(L - 1)):
                    rr = R.metrics(R.blocks(al, m, "RRM"), m)["Lmax"]
                    assert rr == max(min(al[i] + al[i + 1] - 1, h) for i in range(L - 1))


def test_transition_identity():
    for m in range(3, 9):
        for L in range(2, 5):
            for alloc in itertools.product(range(1, m + 1), repeat=L):
                for fam in ("FM", "RRM", "ORRM"):
                    B = R.blocks(list(alloc), m, fam)
                    o = [len(set(B[i - 1]) & set(B[i])) for i in range(1, L)]
                    Z = R.metrics(B, m)["Z"]
                    assert Z == 2 * (2 * sum(alloc) - alloc[-1] - 2 * sum(o))


def test_orrm_path_length_closed_form():
    for m in range(3, 11):
        for L in range(2, 5):
            for alloc in itertools.product(range(1, m + 1), repeat=L):
                al = list(alloc)
                tot = sum(al); Er = 0 if tot <= m else (tot - m) / (L - 1)
                r = [0]
                for i in range(1, L):
                    r.append(max(0, min(int(round(Er)), al[i - 1] - r[-1], al[i])))
                if not all(al[i] + al[i + 1] - r[i + 1] <= m for i in range(L - 1)):
                    continue
                got = R.metrics(R.blocks(al, m, "ORRM"), m)["Lmax"]
                exp = max(min(al[i] + al[i + 1] - r[i + 1] - 1, m // 2) for i in range(L - 1))
                if all(al[i] == 1 and al[i + 1] == 1 and r[i + 1] == 1 for i in range(L - 1)):
                    exp = 0
                assert got == exp, (m, al, r, got, exp)
