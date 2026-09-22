"""
Generates every number, table and figure quoted in the manuscript from results/*.csv.

  python make_tables.py      -> paper/generated/*.tex and paper/figs/*.pdf

No number in the paper text is typed by hand: the text uses the macros written to
paper/generated/results_macros.tex.
"""
from __future__ import annotations
import json, os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GEN = os.path.join(HERE, "paper", "generated")
FIG = os.path.join(HERE, "paper", "figs")
os.makedirs(GEN, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# palette (validated categorical slots 1-4, light mode) + ink
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
NETS = ["NN1", "NN2", "NN3", "NN4", "NN5", "NN6"]
MUS = [1, 8, 16, 32]; LAMS = [8, 64]

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})

macros: dict[str, str] = {}


def mac(name, value):
    macros[name] = value


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def fx(x, nd=2):
    return f"{x:.{nd}f}"


def load():
    d = pd.read_csv(os.path.join(RES, "main.csv"))
    d["alloc_l"] = d["alloc"].map(json.loads)
    return d


def num(x):
    """Readable fixed-point number with thousands separators, 3-4 significant digits."""
    if x >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def idx(df):
    return df.set_index(["net", "mu", "lam", "bwd"]).sort_index()


# ---------------------------------------------------------------------------
def headline(d):
    fm = d[d.family == "FM"]
    cs, mx, f2, bf = (idx(fm[fm.method == k]) for k in ("CS", "MAX", "F200", "BESTFIX"))
    ri = idx(fm[fm.method == "RI"])
    red = lambda base: 1 - cs["T"] / base["T"]
    for c in ("R", "T"):
        s = cs.xs(c, level="bwd").index
        for tag, base in (("Max", mx), ("Two", f2), ("BF", bf)):
            r = red(base).xs(c, level="bwd")
            mac(f"Gain{tag}{c}Min", pct(r.min())); mac(f"Gain{tag}{c}Max", pct(r.max()))
            mac(f"Gain{tag}{c}Mean", pct(r.mean()))
        r = red(mx).xs(c, level="bwd")
        mac(f"GainMax{c}Count", str(int((r > 0.01).sum()))); mac(f"GainMax{c}N", str(len(r)))
        cm = (cs["comm"] / cs["T"]).xs(c, level="bwd")
        mac(f"CommShare{c}Min", pct(cm.min())); mac(f"CommShare{c}Max", pct(cm.max()))
        rg = (ri["T"] / cs["T"] - 1).xs(c, level="bwd")
        mac(f"RIgap{c}Mean", pct(rg.mean(), 1)); mac(f"RIgap{c}Max", pct(rg.max(), 1))
    # R vs T
    R = cs.xs("R", level="bwd"); T = cs.xs("T", level="bwd")
    sp = R["T"] / T["T"]
    mac("TRspeedMin", fx(sp.min(), 1)); mac("TRspeedMax", fx(sp.max(), 1)); mac("TRspeedMed", fx(sp.median(), 1))
    er = T["E_total"] / R["E_total"]
    mac("TRTotLess", pct(max(0.0, 1 - er.min()))); mac("TRTotMore", pct(max(0.0, er.max() - 1)))
    mac("TRTotNMore", str(int((er > 1).sum()))); mac("TRTotN", str(len(er)))
    mac("TRTotMoreMaxMu", str(int(er.idxmax()[1])))
    ne = T["E_net"] / R["E_net"]
    mac("TRNetEnergyMin", pct(1 - ne.max())); mac("TRNetEnergyMax", pct(1 - ne.min()))
    fl = T["flops"] / R["flops"]
    mac("TRflopMin", pct(fl.min() - 1)); mac("TRflopMax", pct(fl.max() - 1))
    sr = T["sram_peak"] / R["sram_peak"]
    mac("TRsramMin", fx(sr.min(), 2)); mac("TRsramMax", fx(sr.max(), 2))
    # the selected hidden-layer count under R at lambda=64 vs Corollary 1
    # families
    fams = d[d.method == "CS"].pivot_table(index=["net", "mu", "lam", "bwd"], columns="family", values="T")
    spread = (fams.max(axis=1) / fams.min(axis=1) - 1)
    mac("FamSpreadMax", pct(spread.max(), 1))
    return cs, mx, f2, bf


# ---------------------------------------------------------------------------
def fig_baselines(d):
    """Iteration-time ratio of each fixed configuration to the selected allocation (FM)."""
    fm = d[d.family == "FM"]
    cs, mx, f2, bf = (idx(fm[fm.method == k]) for k in ("CS", "MAX", "F200", "BESTFIX"))
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 2.95), sharey="row")
    x = np.arange(len(NETS)); w = 0.26
    for r, c in enumerate(("R", "T")):
        for j, (mu, lam) in enumerate([(8, 8), (8, 64), (32, 8), (32, 64)]):
            ax = axes[r, j]
            key = lambda net: (net, mu, lam, c)
            base = np.array([cs.loc[key(nn), "T"] for nn in NETS])
            for k, (df, col, lab, hatch) in enumerate([(bf, C1, "best fixed", ""), (f2, C2, "fixed 200", "//"),
                                                       (mx, C3, "maximum", "..")]):
                v = np.array([df.loc[key(nn), "T"] for nn in NETS]) / base
                ax.bar(x + (k - 1) * w, v, w * 0.9, color=col, edgecolor="white", linewidth=0.4,
                       hatch=hatch, label=lab)
            ax.axhline(1.0, color=INK, lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels([s.replace("NN", "") for s in NETS])
            ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
            ax.set_title(f"{c}: $\\mu$={mu}, $\\lambda_{{\\max}}$={lam}", pad=2)
            for xi, b in zip(x, base):
                lab = f"{b*1e3:.2g}" if b >= 1e-3 else f"{b*1e3:.2f}"
                ax.text(xi, 0.02, "", fontsize=5)
            if j == 0:
                ax.set_ylabel("time / selected")
            if r == 1:
                ax.set_xlabel("network (NN$k$)")
        axes[r, 0].set_ylim(0, None)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_baselines.pdf")); plt.close(fig)


def tab_absolute(d):
    """Absolute iteration time and energy of the selected allocation (FM), R and T."""
    fm = d[(d.family == "FM") & (d.method == "CS")]
    cs = idx(fm)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Iteration time $T_{\mathrm{step}}$ in $\mu$s of the selected allocation (FM, coordinate search).}",
             r"\label{tab:absolute}", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{@{}ll" + "rr" * 6 + r"@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{NN1} & \multicolumn{2}{c}{NN2} & \multicolumn{2}{c}{NN3} & \multicolumn{2}{c}{NN4} & \multicolumn{2}{c}{NN5} & \multicolumn{2}{c}{NN6}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
             r"$\lmax$ & $\mu$ & " + " & ".join([r"\Rc", r"\Tc"] * 6) + r"\\", r"\midrule"]
    for lam in LAMS:
        for mu in MUS:
            cells = []
            for nn in NETS:
                for c in ("R", "T"):
                    r = cs.loc[(nn, mu, lam, c)]
                    cells.append(num(r['T'] * 1e6))
            lines.append(f"{lam} & {mu} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(GEN, "tab_absolute.tex"), "w").write("\n".join(lines))


# ---------------------------------------------------------------------------
def fig_collective(d):
    fm = idx(d[(d.family == "FM") & (d.method == "CS")])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75))
    x = np.arange(len(NETS)); w = 0.2
    cols = [C1, C2, C3, C4]
    for k, mu in enumerate(MUS):
        for a, (lab, f) in enumerate([
                ("speed-up of \\Tc\\ over \\Rc", lambda R, T: R["T"] / T["T"]),
                ("energy \\Tc/\\Rc", lambda R, T: T["E_total"] / R["E_total"]),
                ("peak SRAM \\Tc/\\Rc", lambda R, T: T["sram_peak"] / R["sram_peak"])]):
            vals = []
            for nn in NETS:
                R = fm.loc[(nn, mu, 64, "R")]; T = fm.loc[(nn, mu, 64, "T")]
                vals.append(f(R, T))
            axes[a].bar(x + (k - 1.5) * w, vals, w * 0.9, color=cols[k], edgecolor="white", lw=0.4,
                        label=f"$\\mu$={mu}")
    titles = ["(a) speed-up of T over R", "(b) total energy, T / R", "(c) peak SRAM per core, T / R"]
    for a, ax in enumerate(axes):
        ax.set_xticks(x); ax.set_xticklabels([s.replace("NN", "") for s in NETS])
        ax.set_xlabel("network (NN$k$)"); ax.set_title(titles[a], pad=2)
        ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
        ax.axhline(1.0, color=INK, lw=0.7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    fig.savefig(os.path.join(FIG, "fig_collective.pdf")); plt.close(fig)


# ---------------------------------------------------------------------------
def tab_mapping(d):
    cs = d[d.method == "CS"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Schedules selected for each mapping family under \Rc\ with $\lmax=64$: iteration time, peak SRAM per core, state transitions $Z$, longest activity run $R_{\mathrm{run}}$ (periods) and longest route $L$ (hops).}",
             r"\label{tab:mapping}", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}lrlrrrrr@{}}", r"\toprule",
             r"Net & $\mu$ & Map & $T$ ($\mu$s) & SRAM (MB) & $Z$ & $R_{\mathrm{run}}$ & $L$\\",
             r"\midrule"]
    for nn in ("NN2", "NN4", "NN6"):
        for mu in (8, 32):
            for fam in ("FM", "RRM", "ORRM"):
                r = cs[(cs.net == nn) & (cs.mu == mu) & (cs.lam == 64) & (cs.bwd == "R") & (cs.family == fam)].iloc[0]
                lines.append(f"{nn if (mu == 8 and fam == 'FM') else ''} & {mu if fam == 'FM' else ''} & {fam} & "
                             f"{num(r['T']*1e6)} & {r['sram_peak']/1e6:.2f} & {num(int(r['Z']))} & "
                             f"{int(r['Rrun'])} & {int(r['maxroute'])}\\\\")
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(GEN, "tab_mapping.tex"), "w").write("\n".join(lines))
    # macros: SRAM reduction RRM vs FM under R
    p = cs[cs.bwd == "R"].pivot_table(index=["net", "mu", "lam"], columns="family", values="sram_peak")
    red = 1 - p["RRM"] / p["FM"]
    mac("SramRRMredMax", pct(red.max())); mac("SramRRMredMin", pct(red.min()))
    red2 = 1 - p["ORRM"] / p["FM"]
    mac("SramORRMredMax", pct(red2.max()))
    pz = cs[cs.bwd == "R"].pivot_table(index=["net", "mu", "lam"], columns="family", values="Z")
    mac("ZRRMratioMax", fx((pz["RRM"] / pz["FM"]).max(), 1))
    pe = cs[cs.bwd == "R"].pivot_table(index=["net", "mu", "lam"], columns="family", values="E_net")
    mac("EnetRRMvsFMMax", pct((pe["RRM"] / pe["FM"] - 1).max()))
    # SRAM range overall (FM, R and T)
    fmcs = cs[cs.family == "FM"]
    mac("SramMinKB", f"{fmcs.sram_peak.min()/1e3:.0f}"); mac("SramMaxMB", f"{fmcs.sram_peak.max()/1e6:.1f}")
    lf = cs["local_frac"]
    mac("LocalFracMax", pct(lf.max(), 1))
    tt = cs[(cs.bwd == "T")].pivot_table(index=["net", "mu", "lam"], columns="family", values="Z")
    same = ((tt["FM"] == tt["RRM"]) & (tt["FM"] == tt["ORRM"])).sum()
    mac("TsameSchedCount", str(int(same))); mac("TsameSchedN", str(len(tt)))


# ---------------------------------------------------------------------------
def energy_tradeoff(d):
    fm = d[d.family == "FM"]
    if not (fm.method == "CSE").any():
        print("energy-optimal rows missing; skipping")
        return
    cs, ce = idx(fm[fm.method == "CS"]), idx(fm[fm.method == "CSE"])
    mx, f2 = idx(fm[fm.method == "MAX"]), idx(fm[fm.method == "F200"])
    er = 1 - ce["E_total"] / cs["E_total"]
    tp = ce["T"] / cs["T"] - 1
    for c in ("R", "T"):
        mac(f"CSEsave{c}Min", pct(er.xs(c, level="bwd").min())); mac(f"CSEsave{c}Max", pct(er.xs(c, level="bwd").max()))
        mac(f"CSEslow{c}Min", pct(tp.xs(c, level="bwd").min())); mac(f"CSEslow{c}Max", pct(tp.xs(c, level="bwd").max()))
        emx = 1 - cs["E_total"] / mx["E_total"]
        mac(f"EvsMax{c}Min", pct(emx.xs(c, level="bwd").min())); mac(f"EvsMax{c}Max", pct(emx.xs(c, level="bwd").max()))
        e2 = (cs["E_total"] / f2["E_total"] - 1).xs(c, level="bwd")
        mac(f"EvFixed{c}Less", pct(max(0.0, -e2.min()))); mac(f"EvFixed{c}More", pct(max(0.0, e2.max())))
        mac(f"EvFixed{c}NLess", str(int((e2 < 0).sum()))); mac(f"EvFixed{c}N", str(len(e2)))
    # energy components share (R and T) for the selected allocation
    for c in ("R", "T"):
        x = cs.xs(c, level="bwd")
        mac(f"TuneShare{c}Min", pct((x.E_tune / x.E_total).min())); mac(f"TuneShare{c}Max", pct((x.E_tune / x.E_total).max()))
        mac(f"DynShare{c}Min", pct((x.E_dyn / x.E_total).min())); mac(f"DynShare{c}Max", pct((x.E_dyn / x.E_total).max()))
        mac(f"CompShare{c}Min", pct((x.E_comp / x.E_total).min())); mac(f"CompShare{c}Max", pct((x.E_comp / x.E_total).max()))
    # example: NN2 mu=8 lam=64 R
    a = cs.loc[("NN2", 8, 64, "R")]; b = ce.loc[("NN2", 8, 64, "R")]
    mac("ExCSalloc", a["alloc"].replace(",", ",\\allowbreak "))
    mac("ExCSEalloc", b["alloc"].replace(",", ",\\allowbreak "))
    mac("ExCST", f"{a['T']*1e6:.0f}"); mac("ExCSET", f"{b['T']*1e6:.0f}")
    mac("ExCSE", f"{a['E_total']*1e3:.2f}"); mac("ExCSEE", f"{b['E_total']*1e3:.2f}")


# ---------------------------------------------------------------------------
def solver_quality(d):
    path = os.path.join(RES, "dp.csv")
    fm = d[(d.family.isin(["FM", "RRM"]))]
    cs = fm[fm.method == "CS"].set_index(["net", "mu", "lam", "bwd", "family"])
    ri = idx(d[(d.family == "FM") & (d.method == "RI")])
    have = os.path.exists(path)
    dp = pd.read_csv(path).set_index(["net", "mu", "lam", "bwd", "family"]) if have else None
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Solver quality over 16 settings per network ($\mu\in\{1,8,16,32\}$, $\lmax\in\{8,64\}$, \Rc\ and \Tc). Gaps are relative to the exact FM optimum of the dynamic program (DP): rounded relaxed initialiser (RI) and coordinate search (CS); ``exact'' counts CS runs that reach the DP optimum.}",
             r"\label{tab:solver}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{RI gap (\%)} & \multicolumn{2}{c}{CS gap (\%)} & CS & mean DP\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Net & mean & max & mean & max & exact & (s)\\", r"\midrule"]
    allri, allcs, exact, tot, dpt = [], [], 0, 0, []
    for nn in NETS:
        rg, cg, ex, n, ts = [], [], 0, 0, []
        for mu in MUS:
            for lam in LAMS:
                for c in ("R", "T"):
                    k = (nn, mu, lam, c, "FM")
                    if not have or k not in dp.index:
                        continue
                    Tdp = dp.loc[k, "T"]; ts.append(dp.loc[k, "seconds"])
                    rg.append(ri.loc[(nn, mu, lam, c), "T"] / Tdp - 1)
                    g = cs.loc[k, "T"] / Tdp - 1
                    cg.append(g); ex += int(g <= 1e-9); n += 1
        if n == 0:
            lines.append(f"{nn} & -- & -- & -- & -- & -- & --\\\\"); continue
        allri += rg; allcs += cg; exact += ex; tot += n; dpt += ts
        lines.append(f"{nn} & {100*np.mean(rg):.2f} & {100*np.max(rg):.2f} & {max(0.0, 100*np.mean(cg)):.3f} & "
                     f"{max(0.0, 100*np.max(cg)):.3f} & {ex}/{n} & {np.mean(ts):.0f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(GEN, "tab_solver.tex"), "w").write("\n".join(lines))
    if tot:
        mac("RIgapMean", f"{100*np.mean(allri):.1f}"); mac("RIgapMax", f"{100*np.max(allri):.1f}")
        mac("CSgapMean", f"{max(0.0, 100*np.mean(allcs)):.3f}"); mac("CSgapMax", f"{max(0.0, 100*np.max(allcs)):.2f}")
        mac("CSexact", str(exact)); mac("CSN", str(tot)); mac("DPtimeMax", f"{np.max(dpt):.0f}")
        # RRM DP vs CS
        rr = []
        for k in dp.index:
            if k[4] == "RRM" and k in cs.index:
                rr.append(cs.loc[k, "T"] / dp.loc[k, "T"] - 1)
        mac("CSgapRRMMax", f"{100*np.max(rr):.2f}" if rr else "--"); mac("RRMN", str(len(rr)))
        mac("CSexactRRM", str(int(sum(1 for g in rr if g <= 1e-9))))
    else:
        for k in ("RIgapMean", "RIgapMax", "CSgapMean", "CSgapMax", "CSexact", "CSN", "DPtimeMax",
                  "CSgapRRMMax", "RRMN", "CSexactRRM"):
            mac(k, "--")
    evals = d[(d.method == "CS") & (d.family == "FM")]["evals"]
    mac("CSevalsMax", f"{int(evals.max()):,}".replace(",", "{,}"))
    sw = d[(d.method == "CS") & (d.family == "FM")]["sweeps"]
    mac("CSsweepsMax", str(int(sw.max())))


# ---------------------------------------------------------------------------
def fig_sens():
    if not os.path.exists(os.path.join(RES, "sens.csv")):
        print("sens.csv missing; skipping"); return
    s = pd.read_csv(os.path.join(RES, "sens.csv"))
    s["gmax"] = 1 - s["T"] / s["T_max"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75))
    specs = [("R", 1, C1, "-", "o"), ("R", 32, C1, "--", "s"), ("T", 1, C2, "-", "o"), ("T", 32, C2, "--", "s")]
    # (a) gain over max-core vs D_cfg (NN6, lambda=8)
    for c, mu, col, ls, mk in specs:
        x = s[(s.param == "D_cfg") & (s.net == "NN6") & (s.lam == 8) & (s.bwd == c) & (s.mu == mu)].sort_values("value")
        axes[0].plot(x.value * 1e9, 100 * x.gmax, ls, color=col, marker=mk, ms=3.5, lw=1.2, label=f"{c}, $\\mu$={mu}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("$D_{\\mathrm{cfg}}$ (ns)"); axes[0].set_ylabel("reduction vs. maximum (%)")
    axes[0].set_title("(a) NN6, $\\lambda_{\\max}$=8", pad=2)
    # (b) gain vs C (NN6, lambda=64)
    for c, mu, col, ls, mk in specs:
        x = s[(s.param == "C") & (s.net == "NN6") & (s.lam == 64) & (s.bwd == c) & (s.mu == mu)].sort_values("value")
        axes[1].plot(x.value / 1e9, 100 * x.gmax, ls, color=col, marker=mk, ms=3.5, lw=1.2)
    axes[1].set_xscale("log"); axes[1].set_xlabel("analytical model: sustained $C$ (GFLOP/s)")
    axes[1].set_xticks([1.5, 6, 24]); axes[1].set_xticklabels(["1.5", "6", "24"])
    axes[1].set_title("(b) NN6, $\\lambda_{\\max}$=64, overhead-free core", pad=2)
    # (c) T/R time ratio vs D_cfg, NN2 & NN6, lambda=8, mu=1 and 128
    for nn, col in (("NN2", C3), ("NN6", C4)):
        for mu, ls, mk in ((1, "-", "o"), (32, "--", "s")):
            x = s[(s.param == "D_cfg") & (s.net == nn) & (s.lam == 8) & (s.mu == mu)]
            p = x.pivot_table(index="value", columns="bwd", values="T")
            axes[2].plot(p.index * 1e9, p["T"] / p["R"], ls, color=col, marker=mk, ms=3.5, lw=1.2,
                         label=f"{nn}, $\\mu$={mu}")
    axes[2].set_xscale("log"); axes[2].set_xlabel("$D_{\\mathrm{cfg}}$ (ns)"); axes[2].set_ylabel("$T_{\\mathrm{step}}$(T) / $T_{\\mathrm{step}}$(R)")
    axes[2].set_title("(c) collective ratio, $\\lambda_{\\max}$=8", pad=2); axes[2].set_ylim(0, 1.05)
    axes[2].axhline(1, color=INK, lw=0.6)
    for ax in axes:
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="upper center", ncol=8, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_sens.pdf")); plt.close(fig)
    # macros
    r = s[s.bwd == "R"]
    main = r[r.param.isin(["D_cfg", "B_lam"])]
    mac("SensRgainMin", pct(main.gmax.min())); mac("SensRgainMax", pct(main.gmax.max()))
    ana = r[(r.param == "C") & (r.value >= 6e9)]
    mac("AnaRgainMin", pct(ana.gmax.min())); mac("AnaRgainMax", pct(ana.gmax.max()))
    anat = s[(s.bwd == "T") & (s.param == "C")]
    mac("AnaTgainMax", pct(anat.gmax.max()))
    pa = s[(s.param == "C") & (s.value >= 6e9)].pivot_table(index=["net", "mu", "lam", "value"], columns="bwd", values="T")
    mac("AnaSpMin", fx((pa["R"] / pa["T"]).min(), 1)); mac("AnaSpMax", fx((pa["R"] / pa["T"]).max(), 1))
    slow = r[(r.param == "C") & (r.value == 1.5e9) & (r.lam == 64)]
    mac("SensRslowCgainMax", pct(slow.gmax.max())); mac("SensRslowCgainMin", pct(slow.gmax.min()))
    ser = r[(r.param == "ser2") & (r.lam == 64)]
    mac("SensRserTwoGainMax", pct(ser.gmax.max()))
    ser8 = r[(r.param == "ser2") & (r.lam == 8)]
    mac("SensRserTwoLamEightMin", pct(ser8.gmax.min())); mac("SensRserTwoLamEightMax", pct(ser8.gmax.max()))
    t = s[s.bwd == "T"]
    big = t[(t.param == "D_cfg") & (t.value >= 1e-7) & (t.mu == 1)]
    mac("SensTbigDcfgGainMax", pct(big.gmax.max()))
    small = t[(t.param == "D_cfg") & (t.value <= 1e-8)]
    mac("SensTsmallDcfgGainMax", pct(small.gmax.max()))
    p = s[s.param == "D_cfg"].pivot_table(index=["net", "mu", "lam", "value"], columns="bwd", values="T")
    ratio = p["T"] / p["R"]
    mac("SensTRratioMax", fx(ratio.max(), 2))
    hi = ratio.xs(1e-6, level="value")
    mac("SensTRratioMaxAtOneUs", fx(hi.max(), 2))


# ---------------------------------------------------------------------------
def enoc_table(d):
    path = os.path.join(RES, "enoc.csv")
    if not os.path.exists(path):
        for k in ("EnocBERmin", "EnocBERmax", "EnocBETmin", "EnocBETmax", "EnocRatioRmin", "EnocRatioRmax",
                  "EnocRatioTmin", "EnocRatioTmax", "EnocBEbestMin", "EnocBEbestMax"):
            mac(k, "--")
        open(os.path.join(GEN, "tab_enoc.tex"), "w").write("% ENoC results pending\n")
        return
    e = pd.read_csv(path)
    fm = idx(d[(d.family == "FM") & (d.method == "CS")])
    Bref = 128 * 3.4e9
    grid = sorted(v for v in e.B_e.unique() if v < 1e17)

    def curve(nn, mu, c, tree):
        x = e[(e.net == nn) & (e.mu == mu) & (e.bwd == c) & (e.tree == tree)]
        g = x[x.B_e.isin(grid)].sort_values("B_e")
        inf = x[x.B_e >= 1e17]["T"].iloc[0]
        ref = x[np.isclose(x.B_e, Bref)]["T"].iloc[0]
        return g.B_e.values, g["T"].values, inf, ref

    def breakeven(Bs, Ts, Tinf, target):
        if Tinf > target:
            return math.inf
        Tm = np.minimum.accumulate(Ts)
        ok = np.where(Tm <= target)[0]
        if len(ok) == 0:
            return math.nan  # > max grid
        j = ok[0]
        if j == 0:
            return Bs[0]
        # log interpolation between j-1 and j
        x0, x1 = math.log(Bs[j - 1]), math.log(Bs[j]); y0, y1 = Tm[j - 1], Tm[j]
        f = (y0 - target) / (y0 - y1) if y0 != y1 else 1
        return math.exp(x0 + f * (x1 - x0))

    def fmt(b):
        if math.isinf(b):
            return r"$\infty$"
        if math.isnan(b):
            return r"$>$10\,T"
        if b >= 1e12:
            return f"{b/1e12:.2g}\\,T"
        return f"{b/1e9:.3g}"

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Analytical electrical-mesh reference ($25\times40$ mesh, XY routing, 2 cycles per hop, allocation re-optimised for the mesh). Break-even link bit rate $B_e^{*}$ (Gb/s per link and direction; T\,=\,Tb/s) at which the mesh matches the ONoC's selected iteration time with $\lmax=64$, under ideal tree multicast and under unicast replication; $\infty$ means the mesh cannot match it at any link rate. The last two columns give the ratio of mesh to ONoC time at a 128-bit-per-cycle link (435\,Gb/s), each network using its better collective. Rows show $\mu=1$ and 32; ranges quoted in the text include all four batch sizes.}",
             r"\label{tab:enoc}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}lr rr rr rr rr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{ONoC \Rc\ vs mesh \Rc} & \multicolumn{2}{c}{ONoC \Tc\ vs mesh \Tc} & \multicolumn{2}{c}{best vs best} & \multicolumn{2}{c}{$T_{\mathrm{mesh}}/T_{\mathrm{ONoC}}$ at 435\,Gb/s}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
             r"Net & $\mu$ & tree & unicast & tree & unicast & tree & unicast & tree & unicast\\", r"\midrule"]
    beR, beT, beB, ratios, uratios = [], [], [], [], []
    cols = {}
    for nn in NETS:
        for mu in MUS:
            cells = []
            for c in ("R", "T"):
                To = fm.loc[(nn, mu, 64, c), "T"]
                for tree in (1, 0):
                    Bs, Ts, Tinf, ref = curve(nn, mu, c, tree)
                    b = breakeven(Bs, Ts, Tinf, To)
                    cells.append(fmt(b)); cols.setdefault((c, tree), []).append(b)
                    if tree == 1:
                        (beR if c == "R" else beT).append(b)
            To = min(fm.loc[(nn, mu, 64, "R"), "T"], fm.loc[(nn, mu, 64, "T")]["T"])
            refs = []
            for tree in (1, 0):
                cand = []
                for c in ("R", "T"):
                    Bs, Ts, Tinf, ref = curve(nn, mu, c, tree)
                    cand.append((Bs, Ts, Tinf, ref))
                Bs = cand[0][0]
                Ts = np.minimum(cand[0][1], cand[1][1]); Tinf = min(cand[0][2], cand[1][2])
                b = breakeven(Bs, Ts, Tinf, To); cells.append(fmt(b)); cols.setdefault(("B", tree), []).append(b)
                if tree == 1:
                    beB.append(b)
                refs.append(min(cand[0][3], cand[1][3]))
            ratios.append(refs[0] / To); uratios.append(refs[1] / To)
            cells.append(f"{refs[0]/To:.2f}"); cells.append(f"{refs[1]/To:.2f}")
            if mu in (1, 32):
                lines.append(f"{nn if mu == 1 else ''} & {mu} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(GEN, "tab_enoc.tex"), "w").write("\n".join(lines))
    fin = lambda xs: [x for x in xs if math.isfinite(x)]
    mac("EnocBERmin", f"{min(fin(beR))/1e9:.1f}"); mac("EnocBERmax", f"{max(fin(beR))/1e9:.0f}")
    mac("EnocRatioMin", fx(min(ratios), 3)); mac("EnocRatioMax", fx(max(ratios), 2))
    mac("EnocURatioMin", fx(min(uratios), 2)); mac("EnocURatioMax", fx(max(uratios), 2))
    def unit(b):
        return (b / 1e12, "Tb/s") if b >= 1e12 else (b / 1e9, "Gb/s")

    def rng(vals):
        f = [v for v in vals if not math.isinf(v) and not math.isnan(v)]
        over = any(math.isnan(v) for v in vals)
        lo_v, lo_u = unit(min(f))
        lo_s = f"{lo_v:.1f}" if lo_u == "Tb/s" else f"{lo_v:.0f}"
        if over:
            return f"{lo_s}\\,{lo_u} to more than 10\\,Tb/s"
        hi_v, hi_u = unit(max(f))
        hi_s = f"{hi_v:.1f}" if hi_u == "Tb/s" else f"{hi_v:.0f}"
        if lo_u == hi_u:
            return f"{lo_s}--{hi_s}\\,{hi_u}"
        return f"{lo_s}\\,{lo_u} to {hi_s}\\,{hi_u}"
    for (c, tree), vals in cols.items():
        tag = {"R": "R", "T": "T", "B": "Best"}[c] + ("Tree" if tree else "Uni")
        mac(f"BE{tag}Range", rng(vals))
    f = fin(beB)
    mac("EnocBEbestMin", f"{min(f)/1e9:.0f}" if f else "--"); mac("EnocBEbestMax", f"{max(f)/1e9:.0f}" if f else "--")
    mac("EnocBEbestInf", str(sum(1 for x in beB if math.isinf(x)))); mac("EnocBEbestN", str(len(beB)))
    fT = fin(beT)
    mac("EnocBETmin", f"{min(fT)/1e9:.0f}" if fT else "--"); mac("EnocBETmax", f"{max(fT)/1e9:.0f}" if fT else "--")
    mac("EnocBETinf", str(sum(1 for x in beT if math.isinf(x)))); mac("EnocBETN", str(len(beT)))


# ---------------------------------------------------------------------------
def corollary(d):
    import onocsim as S
    cs = idx(d[(d.family == "FM") & (d.method == "CS")])
    # NN2 layer 3 has width 1000 (index 2 of the allocation)
    for lam, name in ((64, "SixtyFour"), (8, "Eight")):
        vals = sorted({json.loads(cs.loc[("NN2", mu, lam, "R"), "alloc"])[2] for mu in MUS})
        mac(f"CorSel{name}", f"{vals[0]}--{vals[-1]}" if len(vals) > 1 else str(vals[0]))
        cfg = S.Cfg(mu=8, lam=lam, bwd="R")
        pred = math.sqrt(3 * 1000 * lam * cfg.B_lam / (4 * cfg.psi * cfg.C_F))
        mac(f"CorPred{name}", f"{pred:.0f}")
        # relaxed (real-valued) estimate from the trace-calibrated initialiser, over mu
        rr = sorted({round(json.loads(d[(d.family == "FM") & (d.method == "RI") & (d.net == "NN2") & (d.mu == mu) & (d.lam == lam) & (d.bwd == "R")].relaxed_real.iloc[0])[2]) for mu in MUS})
        mac(f"CorRelax{name}", f"{min(rr)}--{max(rr)}" if min(rr) != max(rr) else str(rr[0]))


def ring_size():
    path = os.path.join(RES, "ringsize.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d["gmax"] = 1 - d["T"] / d["T_max"]; d["gbf"] = 1 - d["T"] / d["T_bestfix"]
    for m_, name in ((64, "SixtyFour"), (256, "TwoFiftySix")):
        x = d[d.m == m_]
        mac(f"RS{name}GmaxR", pct(x[x.bwd == "R"].gmax.max(), 1)); mac(f"RS{name}GbfR", pct(x[x.bwd == "R"].gbf.max()))
        mac(f"RS{name}GmaxT", pct(x[x.bwd == "T"].gmax.max()))
        q = x.pivot_table(index=["net", "mu", "lam"], columns="bwd", values="T")
        sp = q["R"] / q["T"]
        mac(f"RS{name}SpMin", fx(sp.min(), 2)); mac(f"RS{name}SpMax", fx(sp.max(), 2))
        mac(f"RS{name}TslowerMax", pct(max(0.0, 1 / sp.min() - 1)))


def trace_stats():
    import onocsim as S
    df = S.load_traces()
    mac("TraceRows", f"{len(df):,}".replace(",", "{,}"))
    mac("TraceRunsMin", str(int(df.runs.min()))); mac("TraceRunsMax", str(int(df.runs.max())))
    # per-call overhead: intercept of a straight-line fit of time against per-core load q <= 20
    icpt = []
    for (net, period, mu), x in df.groupby(["net", "period", "batch"]):
        n = x.p2.iloc[0] if x.phase.iloc[0] == "FP" else x.p1.iloc[0]
        q = -(-n // x.cores.values)
        t = x.t_mean.values
        sel = q <= 20
        if len(np.unique(q[sel])) >= 5:
            A = np.vstack([np.ones(sel.sum()), q[sel]]).T
            coef, *_ = np.linalg.lstsq(A, t[sel], rcond=None)
            icpt.append(coef[0])
    icpt = np.array(icpt) * 1e6
    tot = viol = 0
    for (net, period, mu), x in df.groupby(["net", "period", "batch"]):
        n = x.p2.iloc[0] if x.phase.iloc[0] == "FP" else x.p1.iloc[0]
        t = S.per_load_times(df, net, period, mu, n, monotone=False)
        v = [t[q] for q in sorted(t)]
        for j in range(1, len(v)):
            tot += 1; viol += int(v[j] < 0.9 * v[j - 1])
    mac("TraceViolPct", f"{100*viol/tot:.1f}")
    mac("TraceOverheadMed", f"{np.median(icpt):.0f}")
    mac("TraceOverheadLo", f"{np.percentile(icpt, 10):.0f}"); mac("TraceOverheadHi", f"{np.percentile(icpt, 90):.0f}")


def phys_macros():
    """Physical-layer numbers of Section 3.6 and Table phys, computed by physlayer.py."""
    from dataclasses import replace
    import physlayer as PL
    p = PL.Phys()
    p1 = replace(p, order=1)
    mac("PhysHop", f"{PL.hop_loss(p):.2f}")
    for F, name in ((1, "One"), (2, "Two"), (3, "Three"), (4, "Four")):
        mac(f"PhysIL{name}", f"{PL.insertion_loss(F, F, p):.1f}")
    mac("PhysFmax", str(PL.max_fanout(p)))
    mac("PhysFlossless", str(PL.lossless_fanout(p.budget_db)))
    mac("PhysHsingle", str(PL.max_hops_single(p)))
    mac("PhysOffset", f"{PL.interferer_offset_db(p):.1f}")
    il = PL.insertion_loss(32, 32, p)
    mac("PhysBcastIL", f"{il:.1f}"); mac("PhysBcastLaunch", f"{PL.launch_dbm(il, p):+.1f}")
    mac("PhysLaunchBudget", f"{PL.launch_dbm(p.budget_db, p):.0f}".replace("-", "$-$"))
    for lam, name in ((4, "Four"), (8, "Eight"), (16, "Sixteen"), (32, "ThirtyTwo"), (64, "SixtyFour")):
        mac(f"PhysSNR{name}", f"{PL.snr_db(p, lam):.1f}".replace("-", "$-$"))
        mac(f"PhysSNRFirst{name}", f"{PL.snr_db(p1, lam):.1f}".replace("-", "$-$"))
    mac("PhysMargin", f"{PL.snr_db(p, 8) - p.snr_min_db:.1f}")
    mac("PhysMaxLamWgSec", str(PL.max_wavelengths(p))); mac("PhysMaxLamWgFirst", str(PL.max_wavelengths(p1)))
    mac("PhysFSRSecond", str(math.ceil(PL.fsr_needed(64, p)))); mac("PhysFSRFirst", str(math.ceil(PL.fsr_needed(64, p1))))
    for pitch, name in ((1.25, "Small"), (5.0, "Large")):
        mac(f"PhysFmax{name}Pitch", str(PL.max_fanout(replace(p, pitch_mm=pitch))))
    mac("PhysFmaxOpt", str(PL.max_fanout(replace(p, L_drop=0.1))))
    mac("PhysFmaxSixdB", str(PL.max_fanout(replace(p, budget_db=6.0))))


def cluster_tables():
    """Section 6.8: physically constrained clustered ring."""
    path = os.path.join(RES, "cluster_main.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d["gmax"] = d.T_max / d["T"] - 1; d["gbf"] = d.T_bf / d["T"] - 1
    d["cost"] = d["T"] / d.T_ideal - 1; d["ecost"] = d.E_total / d.E_ideal - 1
    d["Enet"] = d.E_laser + d.E_tune + d.E_dyn + d.E_fab + d.E_leak
    d["commsh"] = d.comm / d["T"]; d["netsh"] = d.Enet / d.E_total
    mac("ClILmax", f"{d.maxIL.max():.2f}"); mac("ClFmax", str(int(d.maxF.max())))
    mac("ClLaunchMax", f"{d.max_launch_dbm.max():.1f}".replace("-", "$-$"))
    mac("ClChainMax", str(int(d.max_chain.max())))
    mac("ClPureMax", str(int(d.pure_relays.max())))
    mac("ClCostMax", pct(d.cost.max(), 1))
    for b in ("R", "T"):
        x = d[d.bwd == b]
        mac(f"ClCostMax{b}", pct(x.cost.max(), 1))
        mac(f"ClEcostMin{b}", pct(x.ecost.min(), 0).replace("-", "$-$")); mac(f"ClEcostMax{b}", pct(x.ecost.max(), 0))
        neg = x[x.ecost < 0]
        mac(f"ClEcostNote{b}", "" if len(neg) == 0 else
            f" (it fell, by at most {pct(-neg.ecost.min(), 0)}\\%, in {len(neg)} of {len(x)} settings)")
        mac(f"ClGainMax{b}", pct(x.gmax.max(), 1))
        mac(f"ClGainBf{b}", pct(x.gbf.max(), 1))
        mac(f"ClComm{b}Min", pct(x.commsh.min(), 0)); mac(f"ClComm{b}Max", pct(x.commsh.max(), 0))
        mac(f"ClNet{b}Min", pct(x.netsh.min(), 0)); mac(f"ClNet{b}Max", pct(x.netsh.max(), 0))
    mac("ClGainMaxRSixtyFour", pct(d[(d.bwd == "R") & (d.Lam == 64)].gmax.max(), 1))
    gt = d[d.bwd == "T"].gmax.max()
    mac("ClGainTPhrase", r"coincided with it in every setting under \Tc" if gt <= 1e-12 else
        f"was at most {pct(gt, 1)}\\% faster under \\Tc")
    mac("ClCostMaxSixtyFour", pct(d[d.Lam == 64].cost.max(), 1))
    mac("ClLaserShareMax", pct((d.E_laser / d.Enet).max(), 1))
    lr = d.E_ideal_budget_laser / d.E_laser
    mac("ClLaserRatioMin", f"{lr.min():.0f}"); mac("ClLaserRatioMax", f"{lr.max():.0f}")
    p = d.pivot_table(index=["net", "mu", "Lam"], columns="bwd", values="T")
    sp = p["R"] / p["T"]
    mac("ClTRMin", fx(sp.min(), 2)); mac("ClTRMax", fx(sp.max(), 2)); mac("ClTRMed", fx(sp.median(), 2))
    mac("ClTRSlowerN", str(int((sp < 1).sum()))); mac("ClTRN", str(len(sp)))
    mac("ClTRSlowerMax", pct(1 / sp.min() - 1, 0))
    slow = sp[sp < 1]
    assert all(mu == 1 and lam == 64 for (_, mu, lam) in slow.index), slow   # stated in the text
    # table: mu = 8
    x = d[d.mu == 8].set_index(["net", "bwd", "Lam"])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Physically constrained clustered ring (64 hubs $\times$ 16 cores, FM, $\mu=8$): iteration time of the exact optimal allocation in $\mu$s for each collective and channel count $\Lambda$ per link, the largest increase over the same ring without physical constraints, and the largest reduction relative to the maximum allocation.}",
             r"\label{tab:cluster}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{\Rc} & \multicolumn{2}{c}{\Tc} & Budget & Alloc.\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Net & $\Lambda=8$ & 64 & 8 & 64 & cost (\%) & gain (\%)\\", r"\midrule"]
    for net in NETS:
        vals = [x.loc[(net, b, L), "T"] * 1e6 for b in ("R", "T") for L in (8, 64)]
        cost = max(x.loc[(net, b, L), "cost"] for b in ("R", "T") for L in (8, 64))
        gain = max(x.loc[(net, b, L), "gmax"] for b in ("R", "T") for L in (8, 64))
        lines.append(f"{net} & " + " & ".join(f"{v:,.0f}".replace(",", "{,}") for v in vals) +
                     f" & {100*cost:.2f} & {100*gain:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(GEN, "tab_cluster.tex"), "w").write("\n".join(lines))
    # conservative losses and a 6 dB budget
    lp = os.path.join(RES, "cluster_loss.csv")
    if os.path.exists(lp):
        l = pd.read_csv(lp)
        base = d[d.mu == 8].set_index(["net", "bwd", "Lam"])["T"]
        l["vsbase"] = [r["T"] / base[(r.net, r.bwd, r.Lam)] - 1 for _, r in l.iterrows()]
        l["cost"] = l["T"] / l.T_ideal - 1
        for v, name in (("drop01", "Opt"), ("budget6", "SixdB")):
            y = l[l.variant == v]
            mac(f"ClLoss{name}Cost", pct(y.cost.max(), 1)); mac(f"ClLoss{name}IL", f"{y.maxIL.max():.1f}")
            mac(f"ClLoss{name}F", str(int(y.maxF.max())))
            mac(f"ClLoss{name}VsBase", pct(y.vsbase.abs().max(), 1))
    # cluster size
    sp_ = os.path.join(RES, "cluster_size.csv")
    if os.path.exists(sp_):
        z = pd.read_csv(sp_)
        z["gmax"] = z.T_max / z["T"] - 1; z["gbf"] = z.T_bf / z["T"] - 1; z["cost"] = z["T"] / z.T_ideal - 1
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Effect of cluster size on the clustered ring (1,024 cores; NN2, NN4, NN6; $\mu\in\{1,8,32\}$; $\Lambda\in\{8,64\}$): largest insertion loss, longest relay chain, largest time cost of the physical limits, and largest reduction of the selected allocation relative to the maximum and best fixed allocations.}",
                 r"\label{tab:clsize}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
                 r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
                 r"Hubs & IL & Chain & Cost & \multicolumn{2}{c}{vs max (\%)} & \multicolumn{2}{c}{vs fixed (\%)}\\",
                 r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
                 r"$\times$ cores & (dB) & & (\%) & \Rc & \Tc & \Rc & \Tc\\", r"\midrule"]
        for (N, c), g in sorted(z.groupby(["N", "c"]), key=lambda t: -t[0][0]):
            gr = g[g.bwd == "R"]; gt = g[g.bwd == "T"]
            lines.append(f"{N}$\\times${c} &{g.maxIL.max():.2f} & {int(g.max_chain.max())} & {100*g.cost.max():.1f} & "
                         f"{100*gr.gmax.max():.1f} & {100*gt.gmax.max():.1f} & {100*gr.gbf.max():.1f} & {100*gt.gbf.max():.1f}\\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(GEN, "tab_clsize.tex"), "w").write("\n".join(lines))
        for (N, c), name in (((256, 4), "Small"), ((16, 64), "Large")):
            g = z[(z.N == N) & (z.c == c)]
            if len(g) == 0:
                continue
            gr = g[g.bwd == "R"]; gt = g[g.bwd == "T"]
            mac(f"ClSize{name}GainR", pct(gr.gmax.max(), 0)); mac(f"ClSize{name}GainT", pct(gt.gmax.max(), 0))
            mac(f"ClSize{name}BfR", pct(gr.gbf.max(), 0)); mac(f"ClSize{name}BfT", pct(gt.gbf.max(), 0))
            mac(f"ClSize{name}Cost", pct(g.cost.max(), 1)); mac(f"ClSize{name}Chain", str(int(g.max_chain.max())))
            for mu_, mname in ((1, "One"), (8, "Eight"), (32, "ThirtyTwo")):
                mac(f"ClSize{name}Cost{mname}", pct(g[g.mu == mu_].cost.max(), 1))
            mac(f"ClSize{name}Faster", pct(max(0.0, -g.cost.min()), 0))
            mac(f"ClSize{name}F", str(int(g.maxF.max())))
            mac(f"ClSize{name}N", str(len(g)))
            mac(f"ClSize{name}MaxOptN", str(int((g.gmax <= 1e-12).sum())))
            q = g.pivot_table(index=["net", "mu", "Lam"], columns="bwd", values="T")
            s2 = q["R"] / q["T"]
            mac(f"ClSize{name}TRMin", fx(s2.min(), 2)); mac(f"ClSize{name}TRMax", fx(s2.max(), 2))
    cp = os.path.join(RES, "cluster_csdp.csv")
    if os.path.exists(cp):
        e = pd.read_csv(cp)
        gap = e.T_cs / e.T_dp - 1
        mac("ClCSexact", str(int((gap <= 1e-9).sum()))); mac("ClCSN", str(len(e)))
        mac("ClCSgapMax", pct(max(0.0, gap.max()), 2))


def hb_phys_macros():
    """Physical-layer numbers of Section 3.6 (Hummingbird-style network), from physhb.py."""
    import physhb as PH
    p = PH.HBPhys()
    mac("HbFsplit", str(PH.f_split(p)))
    mac("HbSplitSix", f"{10 * math.log10(PH.f_split(p)):.1f}")
    mac("HbSplitEight", f"{10 * math.log10(8):.1f}")
    mac("HbOMA", f"{PH.oma_db(p):.2f}")
    for B, nm in ((10e9, "Ten"), (20e9, "Twenty"), (40e9, "Forty")):
        mac(f"HbSens{nm}", f"{PH.sens_dbm(B, p):.1f}".replace("-", "$-$"))
        for P, pn in ((0.0, "Zero"), (8.0, "Eight")):
            mac(f"HbBudget{pn}{nm}", f"{PH.budget_db(B, P, p):.1f}")
            for mux in ("SDM", "WDM"):
                mac(f"HbF{mux.lower().capitalize()}{pn}{nm}", str(PH.f_max(B, P, mux, 16, 2.5, p)))
    mac("HbHop", f"{PH.hop_db(2.5, p):.2f}")
    for F, nm in ((1, "One"), (2, "Two"), (3, "Three"), (6, "Six")):
        mac(f"HbILSdm{nm}", f"{PH.il_db(F, F, 'SDM', 1, 2.5, p):.1f}")
        mac(f"HbILWdm{nm}", f"{PH.il_db(F, F, 'WDM', 16, 2.5, p):.1f}")
    mac("HbGamma", f"{PH.gamma_wdm_db(16, p):.1f}")
    mac("HbThrough", f"{PH.through_db(16, p):.2f}")
    mac("HbPlaneFortySix", f"{PH.lane_power_dbm(40e9, 6, 'SDM', 1, 2.5, p):.1f}")
    far = 32
    il = PH.il_db(far, far, "SDM", 1, 2.5, p)
    mac("HbIdealIL", f"{il:.1f}"); mac("HbIdealP", f"{PH.sens_dbm(40e9, p) + p.margin_db + il:+.1f}")


def hb_tables():
    """Section 6.8: Hummingbird-style network."""
    path = os.path.join(RES, "hb_main.csv")
    if not os.path.exists(path):
        return
    def prep(f):
        d = pd.read_csv(os.path.join(RES, f))
        d["cost"] = d["T"] / d.T_ideal - 1; d["gmax"] = d.T_max / d["T"] - 1; d["gbf"] = d.T_bf / d["T"] - 1
        d["commsh"] = d.comm / d["T"]; d["Enet"] = d.E_laser + d.E_tune + d.E_dyn + d.E_fab + d.E_leak
        return d
    d = prep("hb_main.csv")
    for k, kn in ((1, "One"), (4, "Four")):
        for b in ("R", "T"):
            x = d[(d.k == k) & (d.bwd == b)]
            mac(f"HbCost{b}k{kn}", pct(x.cost.max(), 1))
            mac(f"HbComm{b}k{kn}Min", pct(x.commsh.min(), 0)); mac(f"HbComm{b}k{kn}Max", pct(x.commsh.max(), 0))
        q = d[d.k == k].pivot_table(index=["net", "mu"], columns="bwd", values="T")
        sp = q["R"] / q["T"]
        mac(f"HbTRk{kn}Min", fx(sp.min(), 2)); mac(f"HbTRk{kn}Med", fx(sp.median(), 2)); mac(f"HbTRk{kn}Max", fx(sp.max(), 2))
    mac("HbGainMax", pct(d.gmax.max(), 1)); mac("HbGainBf", pct(d.gbf.max(), 1))
    mac("HbFmain", str(int(d.F.max()))); mac("HbChainMax", str(int(d.max_chain.max())))
    mac("HbPlaneMax", f"{d.P_lane_dbm.max():.1f}")
    for comp, nm in (("E_laser", "Laser"), ("E_fab", "Fab"), ("E_dyn", "Dyn"), ("E_tune", "Tune")):
        s = d[comp] / d.Enet
        mac(f"HbShare{nm}Min", pct(s.min(), 0)); mac(f"HbShare{nm}Max", pct(s.max(), 0))
    e = d.pivot_table(index=["net", "mu", "k"], columns="bwd", values="E_total")
    er = e["T"] / e["R"]
    mac("HbETRMin", fx(er.min(), 2)); mac("HbETRMax", fx(er.max(), 2))
    # table: mu = 8
    x = d[d.mu == 8].set_index(["net", "k", "bwd"])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Hummingbird-style network (64 clusters $\times$ 16 cores, FM, $\mu=8$, 40\,Gb/s lanes, $F=" + str(int(d.F.max())) + r"$): iteration time of the exact optimal allocation in $\mu$s with $k$ lanes per hub and direction, the largest increase over the same network without the fan-out limit, and the speed-up of \Tc\ over \Rc\ with one lane.}",
             r"\label{tab:hb}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{$k=1$} & \multicolumn{2}{c}{$k=4$} & Limit & \Tc\ vs \Rc\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Net & \Rc & \Tc & \Rc & \Tc & cost (\%) & ($k=1$)\\", r"\midrule"]
    for net in NETS:
        vals = [x.loc[(net, k, b), "T"] * 1e6 for k in (1, 4) for b in ("R", "T")]
        cost = max(x.loc[(net, k, b), "cost"] for k in (1, 4) for b in ("R", "T"))
        sp = x.loc[(net, 1, "R"), "T"] / x.loc[(net, 1, "T"), "T"]
        lines.append(f"{net} & " + " & ".join(f"{v:,.0f}".replace(",", "{,}") for v in vals) +
                     f" & {100*cost:.1f} & {sp:.2f}$\\times$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(GEN, "tab_hb.tex"), "w").write("\n".join(lines))
    # lanes
    lp = os.path.join(RES, "hb_lanes.csv")
    if os.path.exists(lp):
        l = prep("hb_lanes.csv")
        for k, kn in ((1, "One"), (2, "Two"), (4, "Four"), (10, "Ten")):
            y = l[l.k == k]
            mac(f"HbLanes{kn}CostR", pct(y[y.bwd == "R"].cost.max(), 1))
            mac(f"HbLanes{kn}CostT", pct(y[y.bwd == "T"].cost.max(), 1))
    # rate and laser class
    rp = os.path.join(RES, "hb_rate.csv")
    if os.path.exists(rp):
        r = prep("hb_rate.csv"); r["Bg"] = (r.B / 1e9).round().astype(int)
        base = r[r.Bg == 40].set_index(["net", "mu", "bwd", "P_lane_max", "mux"])
        r["rel"] = [row["T"] / base.loc[(row.net, row.mu, row.bwd, row.P_lane_max, row.mux), "T"] for _, row in r.iterrows()]
        r["relE"] = [row["Enet"] / base.loc[(row.net, row.mu, row.bwd, row.P_lane_max, row.mux), "Enet"] for _, row in r.iterrows()]
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Lane rate at a fixed 40\,Gb/s per hub and direction (64 clusters; NN2, NN4, NN6; $\mu\in\{1,8,32\}$): receivers per transmission $F$ and the \Rc\ and \Tc\ iteration times relative to one 40\,Gb/s lane, for comb-class (0\,dBm) and DFB-class (8\,dBm) lanes, one wavelength per waveguide (SDM) or 16 per waveguide (WDM).}",
                 r"\label{tab:hbrate}", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
                 r"\begin{tabular}{@{}llrccrcc@{}}", r"\toprule",
                 r" & & \multicolumn{3}{c}{0\,dBm lanes} & \multicolumn{3}{c}{8\,dBm lanes}\\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                 r" & Lanes & $F$ & \Rc & \Tc & $F$ & \Rc & \Tc\\", r"\midrule"]
        for mux in ("SDM", "WDM"):
            for Bg, k in ((10, 4), (20, 2), (40, 1)):
                cells = [mux if Bg == 10 else "", f"{k}$\\times${Bg}\\,Gb/s"]
                for P in (0.0, 8.0):
                    y = r[(r.mux == mux) & (r.Bg == Bg) & (r.P_lane_max == P)]
                    cells.append(str(int(y.F.max())))
                    for b in ("R", "T"):
                        z = y[y.bwd == b].rel
                        cells.append(f"{z.min():.2f}" if abs(z.max() - z.min()) < 0.005 else f"{z.min():.2f}--{z.max():.2f}")
                lines.append(" & ".join(cells) + r"\\")
            if mux == "SDM":
                lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(GEN, "tab_hbrate.tex"), "w").write("\n".join(lines))
        for mux in ("SDM", "WDM"):
            y = r[(r.mux == mux) & (r.P_lane_max == 0.0) & (r.Bg < 40)]
            mac(f"HbRate{mux.capitalize()}GainRMin", pct(1 - y[y.bwd == "R"].rel.max(), 0))
            mac(f"HbRate{mux.capitalize()}GainRMax", pct(1 - y[y.bwd == "R"].rel.min(), 0))
            mac(f"HbRate{mux.capitalize()}ERelMin", fx(y.relE.min(), 2)); mac(f"HbRate{mux.capitalize()}ERelMax", fx(y.relE.max(), 2))
            z = r[(r.mux == mux) & (r.P_lane_max == 0.0) & (r.Bg == 40)]
            mac(f"HbRate{mux.capitalize()}CostForty", pct(z.cost.max(), 0))
        y8 = r[(r.P_lane_max == 8.0)]
        mac("HbRateEightDiff", pct((y8.rel - 1).abs().max(), 1))
        ys = r[r.F <= 3]
        ys = ys.assign(g=ys.T_max / ys["T"] - 1, gb=ys.T_bf / ys["T"] - 1)
        mac("HbRateAllocGain", pct(ys.g.max(), 1)); mac("HbRateAllocBf", pct(ys.gb.max(), 1))
    cp = os.path.join(RES, "hb_clusters.csv")
    if os.path.exists(cp):
        c = prep("hb_clusters.csv")
        for C, nm in ((8, "Eight"), (16, "Sixteen"), (32, "ThirtyTwo"), (64, "SixtyFour")):
            y = c[c.C == C]
            mac(f"HbCl{nm}Cost", pct(y.cost.max(), 1)); mac(f"HbCl{nm}Chain", str(int(y.max_chain.max())))
            q = y.pivot_table(index=["net", "mu"], columns="bwd", values="T")
            s = q["R"] / q["T"]
            mac(f"HbCl{nm}TRMin", fx(s.min(), 2)); mac(f"HbCl{nm}TRMax", fx(s.max(), 2))
        mac("HbClGainMax", pct(c.gmax.max(), 1))
    dp_ = os.path.join(RES, "hb_dcfg.csv")
    if os.path.exists(dp_):
        g = pd.read_csv(dp_)
        key = ["net", "mu", "bwd", "k"]
        base = g[g.D_cfg == 1e-8].set_index(key)["T"]
        g = g.join(base.rename("T10"), on=key)
        slow = g[g.D_cfg == 1e-6]
        mac("HbDcfgCost", pct((slow["T"] / slow.T10 - 1).max(), 1))
        mac("HbDcfgGmax", pct((g.T_max / g["T"] - 1).max(), 1))
        q = g.pivot_table(index=["net", "mu", "k", "D_cfg"], columns="bwd", values="T")
        rt = (q["R"] / q["T"]).unstack("D_cfg")
        mac("HbDcfgTRShift", fx(rt.sub(rt[1e-8], axis=0).abs().max().max(), 2))


def write_highlights():
    hl = [
        "Ownership-based model of FCNN training on a wavelength-slotted optical ring",
        "Hummingbird-style optical broadcast keeps every split within 8 dB via relays",
        "Exact dynamic program for fixed and round-robin mappings validates the search",
        f"Transpose-broadcast backpropagation is {macros['TRspeedMin']}-{macros['TRspeedMax']}x faster than error reduction",
        "Optical advantage over a wide-link mesh hinges on the backward collective",
    ]
    for h in hl:
        assert len(h) <= 85, (len(h), h)
    with open(os.path.join(GEN, "highlights.txt"), "w") as f:
        f.write("\n".join("- " + h for h in hl) + "\n")


def write_macros():
    with open(os.path.join(GEN, "results_macros.tex"), "w") as f:
        f.write("% generated by make_tables.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    d = load()
    headline(d)
    fig_baselines(d)
    tab_absolute(d)
    fig_collective(d)
    tab_mapping(d)
    energy_tradeoff(d)
    solver_quality(d)
    fig_sens()
    enoc_table(d)
    corollary(d)
    ring_size()
    trace_stats()
    hb_phys_macros()
    hb_tables()
    write_macros()
    write_highlights()
    print(len(macros), "macros")
