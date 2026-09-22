"""
Physical layer of the Hummingbird-style broadcast network (Section 3.6).

Every hub owns k transmit lanes per ring direction. A lane is single-writer: its light is
tapped by the receivers of the next F hubs downstream (fixed taps, as in Hummingbird's
U-shaped broadcast waveguides), and the F-th hub can regenerate and forward the data.

Lanes are either
  SDM: one wavelength per waveguide, one waveguide per lane (Hummingbird); receivers are
       photodiodes behind taps, so there are no drop filters and no inter-channel crosstalk;
  WDM: lam_wg wavelengths of one writer share a waveguide; receivers demultiplex them with
       second-order microring filters.

Worst-case loss of one transmission reaching F receivers over H hops (dB):
  IL(F, H) = L_mod + L_oma + F*L_tap + 10*log10(F) + [L_drop + L_thr]_WDM + H*(alpha*d_hop + L_bend)
where L_oma converts average power to optical modulation amplitude (OMA) for the extinction
ratio, and L_thr is the off-resonance loss of the writer's other modulators (WDM only).

Two constraints:
  * splitting loss 10*log10(F) <= 8 dB (a device-independent limit; the split is the part of the
    loss that no device improvement removes)  ->  F <= 6;
  * link budget  P_lane - IL(F, F) >= S(B) + margin, with the OMA sensitivity S(B) scaled from
    a measured receiver by 15 dB per decade of bit rate (thermal-noise-limited front end) and
    the on-chip lane power P_lane limited by the laser class.

Crosstalk (WDM only): all wavelengths of one waveguide come from one writer and travel the
same path, so they arrive with equal power; the worst-case signal-to-crosstalk ratio uses the
measured adjacent-channel crosstalk of second-order filters at 200 GHz and their fourth-power
roll-off, plus the residual of a reused wavelength, and must reach 16.9 dB (OOK, BER 1e-12).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HBPhys:
    alpha_db_cm: float = 1.0      # silicon waveguide (200/300 mm foundry platforms)
    L_bend: float = 0.005         # one bend per hop
    L_mod: float = 2.5            # microdisk modulator insertion loss at maximum OMA (Daudlin 2025)
    ER_db: float = 4.0            # extinction ratio (Daudlin 2025)
    Q_mod: float = 5800.0         # modulator Q, for off-resonance loss of neighbouring modulators
    L_tap: float = 0.2            # excess loss of each tap splitter
    L_drop: float = 1.0           # second-order drop filter (WDM only; 0.3-2.9 dB in Yu 2026)
    fsr_nm: float = 25.6          # microdisk FSR (Daudlin 2025): 16 channels at 200 GHz
    lambda0_nm: float = 1550.0
    split_max_db: float = 8.0     # limit on the splitting loss
    S10_dbm: float = -22.3        # receiver OMA sensitivity at 10 Gb/s (Daudlin 2025)
    S_slope: float = 15.0         # dB per decade of bit rate
    margin_db: float = 3.0        # system margin (BER target, drift, ageing)
    xt_adj_db: float = -22.1      # adjacent-channel crosstalk, second-order filter, 200 GHz (Yu 2026)
    K_res_db: float = -35.0       # residual of a reused wavelength (Pintus 2013)
    snr_min_db: float = 16.9      # OOK, BER 1e-12 (Q-factor 7)


def sens_dbm(B: float, p: HBPhys = HBPhys()) -> float:
    return p.S10_dbm + p.S_slope * math.log10(B / 10e9)


def oma_db(p: HBPhys = HBPhys()) -> float:
    """Average-to-OMA conversion: OMA = P_avg * 2 (r - 1) / (r + 1)."""
    r = 10 ** (p.ER_db / 10)
    return 10 * math.log10((r + 1) / (2 * (r - 1)))


def through_db(lam_wg: int, p: HBPhys = HBPhys()) -> float:
    """Off-resonance loss a channel sees passing the writer's other modulators (Lorentzian dip
    of depth 1 - 1/ER at the channel's detuning)."""
    if lam_wg <= 1:
        return 0.0
    sp = p.fsr_nm / lam_wg
    w = p.lambda0_nm / p.Q_mod
    depth = 1 - 10 ** (-p.ER_db / 10)
    s = 0.0
    for k in range(1, lam_wg):
        d = min(k, lam_wg - k) * sp
        s += -10 * math.log10(1 - depth / (1 + (2 * d / w) ** 2))
    return s


def hop_db(pitch_mm: float, p: HBPhys = HBPhys()) -> float:
    return p.alpha_db_cm * pitch_mm / 10.0 + p.L_bend


def il_db(F: int, H: int, mux: str, lam_wg: int, pitch_mm: float, p: HBPhys = HBPhys()) -> float:
    L = p.L_mod + oma_db(p) + F * p.L_tap + 10 * math.log10(F) + H * hop_db(pitch_mm, p)
    if mux == "WDM":
        L += p.L_drop + through_db(lam_wg, p)
    return L


def f_split(p: HBPhys = HBPhys()) -> int:
    return int(math.floor(10 ** (p.split_max_db / 10) + 1e-9))


def budget_db(B: float, P_lane_dbm: float, p: HBPhys = HBPhys()) -> float:
    return P_lane_dbm - sens_dbm(B, p) - p.margin_db


def f_budget(B, P_lane_dbm, mux, lam_wg, pitch_mm, p: HBPhys = HBPhys(), cap=4096) -> int:
    bud = budget_db(B, P_lane_dbm, p)
    F = 0
    while F < cap and il_db(F + 1, F + 1, mux, lam_wg, pitch_mm, p) <= bud + 1e-12:
        F += 1
    return F


def f_max(B, P_lane_dbm, mux, lam_wg, pitch_mm, p: HBPhys = HBPhys()) -> int:
    return min(f_split(p), f_budget(B, P_lane_dbm, mux, lam_wg, pitch_mm, p))


def lane_power_dbm(B, F, mux, lam_wg, pitch_mm, p: HBPhys = HBPhys()) -> float:
    """Launch power per lane that just closes the budget for a window of F hops."""
    return sens_dbm(B, p) + p.margin_db + il_db(F, F, mux, lam_wg, pitch_mm, p)


def gamma_wdm_db(lam_wg: int, p: HBPhys = HBPhys()) -> float:
    """Worst-case signal-to-crosstalk ratio of a WDM receiver (equal-power channels)."""
    if lam_wg <= 1:
        return float("inf")
    k1 = 10 ** (p.xt_adj_db / 10)
    s = 0.0
    for k in range(1, lam_wg):
        d = min(k, lam_wg - k)
        s += k1 / d ** 4          # second-order roll-off beyond the adjacent channel
    s += 10 ** (p.K_res_db / 10)
    return -10 * math.log10(s)


def pitch_for(C: int, die_mm: float = 20.0) -> float:
    return die_mm / math.sqrt(C)


if __name__ == "__main__":
    p = HBPhys()
    print("OMA factor", round(oma_db(p), 2), "dB; through(16)", round(through_db(16, p), 3), "dB; F_split", f_split(p))
    for B in (10e9, 25e9, 40e9):
        print(f"B={B/1e9:.0f}G  S={sens_dbm(B, p):.1f} dBm", end="  ")
        for P in (0.0, 8.0):
            for mux, lam in (("SDM", 1), ("WDM", 16)):
                print(f"[{mux} {P:.0f}dBm: budget {budget_db(B, P, p):.1f} F_b {f_budget(B, P, mux, lam, 2.5, p)} "
                      f"F {f_max(B, P, mux, lam, 2.5, p)}]", end=" ")
        print()
    for F in (1, 2, 4, 6):
        print("IL SDM", F, round(il_db(F, F, "SDM", 1, 2.5, p), 2), " WDM", round(il_db(F, F, "WDM", 16, 2.5, p), 2))
    print("gamma WDM 8/16/32:", [round(gamma_wdm_db(l, p), 1) for l in (8, 16, 32)])
    print("Hummingbird-like single broadcast to 7 clusters would split", round(10 * math.log10(7), 2), "dB")
