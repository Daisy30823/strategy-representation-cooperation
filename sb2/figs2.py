"""Generate the revised manuscript figures from a reproduction workspace."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from figstyle import apply_figure_style, panel_letter

COL = {"TAB1": "#8c8c8c", "TAB2": "#222222", "CIRC16": "#1f6fb4", "PROG16": "#e07b1a", "AUTO4": "#2e9e5b", "RNN3": "#8e44ad",
       "CIRC1": "#1f6fb4", "PROG1": "#e07b1a", "PERC1": "#b8860b", "RNN1t": "#8e44ad", "AUTO2": "#2e9e5b"}
LAB = {"TAB1": "Table-1", "TAB2": "Table-2", "CIRC16": "Circuit", "PROG16": "Program", "AUTO4": "Automaton", "RNN3": "Recurrent net",
       "CIRC1": "Circuit (2 gates)", "PROG1": "Program (4 instr.)", "PERC1": "Perceptron", "RNN1t": "Recurrent (1 unit)", "AUTO2": "Automaton (k=2)"}
MAINF = ["TAB1", "TAB2", "CIRC16", "PROG16", "AUTO4", "RNN3"]
EXF = ["TAB1", "PERC1", "CIRC1", "PROG1", "RNN1t", "AUTO2"]

def end_labels(ax, x, ys, labels, colors, gap=0.04, dx=0.12, fs=6):
    order = np.argsort(ys); placed = []
    for i in order:
        y = ys[i]
        if placed and y < placed[-1] + gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.text(x + dx, y, labels[i], color=colors[i], fontsize=fs, va="center")


def _load(workspace):
    results = workspace / "res3"
    with (results / "numbers2.json").open(encoding="utf-8") as stream:
        numbers = json.load(stream)
    with (results / "plot_data.json").open(encoding="utf-8") as stream:
        plot = json.load(stream)
    return numbers, {
        "P1": pd.DataFrame(plot["P1"]),
        "G": pd.DataFrame(plot["G"]).set_index(["delta", "b"]),
        "workspace": workspace,
    }


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1(V, D, out):
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.55)
    axA = fig.add_subplot(gs[0, :2]); axB = fig.add_subplot(gs[0, 2:])
    # A: schematic
    axA.axis("off")
    boxes = [(0.13, "Genotype\n(table, circuit,\nprogram, automaton,\nnetwork)"), (0.50, "Compiled\nstrategy"), (0.87, "Phenotype\n(minimal\nmachine)")]
    for x, t in boxes:
        axA.text(x, 0.62, t, ha="center", va="center", fontsize=6.3,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec="#999999", lw=0.6), transform=axA.transAxes)
    for x0, x1 in [(0.30, 0.40), (0.60, 0.72)]:
        axA.annotate("", xy=(x1, 0.62), xytext=(x0, 0.62), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.8))
    axA.text(0.13, 0.12, "point mutation\nacts here", ha="center", fontsize=6.3, transform=axA.transAxes, color="#1f6fb4")
    axA.text(0.80, 0.12, "selection (payoffs)\nacts here", ha="center", fontsize=6.3, transform=axA.transAxes, color="#b03030")
    axA.text(0.5, -0.12, "fixed: game, errors, population dynamics, mutation scheme", ha="center", fontsize=6.3, transform=axA.transAxes)
    panel_letter(axA, "A", dx=-0.02, case="upper")
    # B: mass by number of states
    m2 = {int(k): v for k, v in V["tab2_mass_by_n"].items()}
    m1 = {int(k): v for k, v in V["tab1_exact"]["mass_by_n"].items() if v > 0}
    series = {"TAB1": m1, "TAB2": m2}
    for v in ["CIRC16", "PROG16", "AUTO4", "RNN3"]:
        s = V["samp"].get("samp1e7|" + v) or V["samp"].get("samp3e6|" + v)
        series[v] = {int(k): x for k, x in s["mass_by_n"].items() if x > 0}
    for v in MAINF:
        ks = sorted(series[v]); axB.plot(ks, [series[v][k] for k in ks], "-o", color=COL[v], ms=2.8, lw=1.1, label=LAB[v])
    axB.set_yscale("log"); axB.set_xlabel("States of the minimal machine"); axB.set_ylabel("Probability mass")
    axB.set_xticks(range(1, 10)); axB.set_ylim(1e-8, 1.5)
    axB.legend(frameon=False, ncol=2, loc="lower right", fontsize=6, handlelength=1.2, columnspacing=0.8)
    axB.set_title("Probability mass by machine size", loc="left", fontsize=7.5)
    panel_letter(axB, "B", case="upper")
    # C: probability vs LZ complexity, four encodings
    for i, v in enumerate(["CIRC16", "PROG16", "AUTO4", "RNN3"]):
        ax = fig.add_subplot(gs[1, i])
        f = D["workspace"] / "runB" / ("phen_samp1e7__%s.npz" % v)
        z = np.load(f); cnt = z["cnt"].astype(float); M = float(z["M"]); K = z["klz"]; p = cnt / M
        rng = np.random.default_rng(1)
        idx = np.argsort(-cnt)[:3000]; rest = np.setdiff1d(np.arange(len(cnt)), idx)
        idx = np.concatenate([idx, rng.choice(rest, min(6000, len(rest)), replace=False)]) if len(rest) else idx
        ax.scatter(K[idx], np.log10(p[idx]), s=1.2, color=COL[v], alpha=0.25, lw=0, rasterized=True)
        env = V["samp"]["samp1e7|" + v]["envelope"]; ks = sorted(int(k) for k in env)
        ax.plot(ks, [env[str(k)] for k in ks], color="black", lw=0.8)
        ax.axhline(-21 * np.log10(2), color="#555555", ls="--", lw=0.6)
        ax.set_xlabel("LZ complexity $K$"); ax.set_ylim(-7.4, 0.2)
        ax.set_title(LAB[v], fontsize=7, loc="left", color=COL[v])
        if i == 0:
            ax.set_ylabel(r"$\log_{10}P_E(x)$"); panel_letter(ax, "C", dx=-0.42, case="upper")
        else:
            ax.set_yticklabels([])
    _save(fig, out, "Fig1")


def fig2(V, D, out):
    ex = V["exact"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.32))
    betas = ["0", "0.1", "0.3", "1", "3"]; xs = np.arange(len(betas))
    for v in EXF:
        if v not in ex:
            continue
        y = [ex[v]["dyn"]["point_b" + b] for b in betas]
        axA.plot(xs, y, marker="o", color=COL[v], ms=3, lw=1.1, ls="--" if v in ("PERC1", "PROG1") else "-")
    vv = [v for v in EXF if v in ex]
    end_labels(axA, xs[-1], [ex[v]["dyn"]["point_b3"] for v in vv], [LAB[v] for v in vv], [COL[v] for v in vv])
    axA.set_xticks(xs); axA.set_xticklabels(betas); axA.set_xlim(-0.2, len(betas) + 0.9)
    axA.set_xlabel(r"Selection intensity $\beta$"); axA.set_ylabel("Long-run cooperation (exact)")
    axA.set_ylim(0, 0.95); axA.set_title("Encoding priors persist under selection", loc="left", fontsize=7.5)
    panel_letter(axA, "A", case="upper")
    modes = [("point_b1", "Point mutation", "full"), ("hoc_b1", "Mutants drawn from prior", "none"), ("point_mh_b1", "Point, prior reweighted", "left")]
    w = 0.26; vs = [v for v in EXF if v in ex]
    for j, (k, lab, fs) in enumerate(modes):
        for i, v in enumerate(vs):
            y = ex[v]["dyn"][k]
            axB.bar(i + (j - 1) * w, y, w * 0.92, color=COL[v] if j == 0 else "white", edgecolor=COL[v], lw=1.0,
                    hatch=None if j < 2 else "////")
    SH = {"TAB1": "Table-1", "PERC1": "Percep-\ntron", "CIRC1": "Circuit\n(2 gates)", "PROG1": "Program\n(4 instr.)", "RNN1t": "Network\n(1 unit)", "AUTO2": "Automaton\n(k=2)"}
    axB.set_xticks(range(len(vs))); axB.set_xticklabels([SH[v] for v in vs], fontsize=5.8)
    axB.set_ylabel(r"Long-run cooperation at $\beta=1$")
    axB.set_ylim(0, 0.95)
    hs = [plt.Rectangle((0, 0), 1, 1, fc="#777777", ec="#777777"), plt.Rectangle((0, 0), 1, 1, fc="white", ec="#777777"),
          plt.Rectangle((0, 0), 1, 1, fc="white", ec="#777777", hatch="////")]
    axB.legend(hs, [m[1] for m in modes], frameon=False, fontsize=6, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8)
    axB.set_title("Prior and mutational neighbourhoods both matter", loc="left", fontsize=7.5, pad=12)
    panel_letter(axB, "B", case="upper")
    _save(fig, out, "Fig2")


def fig3(V, D, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.35], wspace=0.5))
    axA, axB, axC = axs
    for i, v in enumerate(MAINF):
        m, h = V["base"][v]
        axA.bar(i, m, 0.62, color=COL[v], yerr=h, capsize=1.5, error_kw=dict(lw=0.7))
        axA.plot(i, V["hoc"][v]["1.0"], "o", mfc="white", mec=COL[v], ms=5, mew=1.1)
        axA.plot(i, V["prior_main"][v], "D", color="black", ms=3)
    axA.set_xticks(range(6)); axA.set_xticklabels([LAB[v].replace(" net", "\nnet") for v in MAINF], rotation=40, ha="right", fontsize=6)
    axA.set_ylabel("Long-run cooperation"); axA.set_ylim(0, 0.97)
    axA.legend([Line2D([], [], color="#777777", lw=5), Line2D([], [], marker="o", mfc="white", mec="#555555", ls=""),
                Line2D([], [], marker="D", color="black", ls="", ms=3)], ["point", "from prior", "prior"],
               frameon=False, fontsize=5.8, loc="upper center", ncol=3, handlelength=0.9, columnspacing=0.6, bbox_to_anchor=(0.58, 1.0))
    axA.set_title("Encodings differ", loc="left", fontsize=7.5)
    panel_letter(axA, "A", dx=-0.25, case="upper")
    P1 = D["P1"]
    bl = ["beta0", "beta0.01", "beta0.1", "beta0.3", "base", "beta3"]; xl = ["0", "0.01", "0.1", "0.3", "1", "3"]
    for v in MAINF:
        y = [P1.loc[v, c] if c != "base" else V["base"][v][0] for c in bl]
        axB.plot(range(len(bl)), y, "-o", color=COL[v], ms=2.8, lw=1.0)
    end_labels(axB, len(bl) - 1, [P1.loc[v, "beta3"] for v in MAINF], [LAB[v] for v in MAINF], [COL[v] for v in MAINF], gap=0.045, dx=0.15, fs=5.5)
    axB.set_xlim(-0.3, len(bl) + 1.1)
    axB.set_xticks(range(len(bl))); axB.set_xticklabels(xl); axB.set_xlabel(r"Selection intensity $\beta$")
    axB.set_ylim(0, 0.85); axB.set_title("Selection widens the gap", loc="left", fontsize=7.5)
    panel_letter(axB, "B", dx=-0.2, case="upper")
    pairs = [("CIRC16", "CIRC16C", "Circuit:\nhistory C"), ("PROG16", "PROG16C", "Program:\nhistory C,\nempty C"),
             ("PROG16", "PROG16dC", "Program:\nempty C"), ("RNN3", "RNN3C", "Network:\nfires at 0")]
    for i, (s, c, lab) in enumerate(pairs):
        axC.bar(i - 0.18, V["base"][s][0], 0.34, color="#cccccc", yerr=V["base"][s][1], capsize=1.5, error_kw=dict(lw=0.7))
        m, h = V["valence"][c]["1.0"]
        axC.bar(i + 0.18, m, 0.34, color=COL[s], yerr=h, capsize=1.5, error_kw=dict(lw=0.7))
    axC.axhline(V["base"]["TAB2"][0], color=COL["TAB2"], ls=":", lw=0.8); axC.text(3.45, V["base"]["TAB2"][0] + 0.015, "Table-2", fontsize=6, ha="right")
    axC.set_xticks(range(4)); axC.set_xticklabels([p[2] for p in pairs], fontsize=5.8)
    axC.set_ylim(0, 0.85); axC.set_title("Conventions set the direction", loc="left", fontsize=7.5)
    axC.legend([Line2D([], [], color="#cccccc", lw=5), Line2D([], [], color="#555555", lw=5)], ["standard", "control"], frameon=False,
               fontsize=6, loc="upper left", handlelength=1.0)
    panel_letter(axC, "C", dx=-0.15, case="upper")
    _save(fig, out, "Fig3")


def fig4(V, D, out):
    G = D["G"]
    fig = plt.figure(figsize=(7.0, 2.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.12)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for j, b in enumerate([1.5, 3.0, 5.0]):
        ax = axs[j]
        sub = G.xs(b, level="b")
        for v in MAINF:
            ax.plot(sub.index, sub[v], "-o", color=COL[v], ms=2.8, lw=1.0)
        ax.axvline(1 / b, color="#777777", ls=":", lw=0.8)
        ax.set_ylim(-0.02, 0.9); ax.set_xlim(0.25, 0.95); ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.set_title("$b/c=%g$" % b, loc="left", fontsize=7.5)
        if j == 1:
            ax.set_xlabel(r"Continuation probability $\delta$")
        if j == 0:
            ax.set_ylabel("Long-run cooperation"); panel_letter(ax, "A", dx=-0.3, case="upper")
        else:
            ax.set_yticklabels([])
        if j == 2:
            for v in MAINF:
                ax.text(0.93, sub[v].iloc[-1], "", color=COL[v])
    axs[0].legend([Line2D([], [], color=COL[v], marker="o", ms=2.5) for v in MAINF], [LAB[v] for v in MAINF], frameon=False, fontsize=5.8,
                  loc="upper left", handlelength=1.2)
    ax = fig.add_subplot(gs[0, 3])
    g = V["grid"]
    cats = ["full grid", r"$\delta>c/b+0.2$"]
    enc = [g["share_enc"], g["share_enc_coopregime"]]; st = [g["share_set"], g["share_set_coopregime"]]
    it = [g["share_int"], 1 - g["share_enc_coopregime"] - g["share_set_coopregime"]]
    ax.bar(range(2), enc, 0.55, color="#1f6fb4", label="encoding")
    ax.bar(range(2), st, 0.55, bottom=enc, color="#bbbbbb", label="game setting")
    ax.bar(range(2), it, 0.55, bottom=np.add(enc, st), color="#eeeeee", edgecolor="#bbbbbb", lw=0.5, label="interaction")
    ax.set_xticks(range(2)); ax.set_xticklabels(cats, fontsize=6); ax.set_ylim(0, 1.0); ax.set_ylabel("Share of variance")
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1, handlelength=1.0)
    panel_letter(ax, "B", dx=-0.1, case="upper")
    _save(fig, out, "Fig4")


def fig5(V, D, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3], wspace=0.42))
    axA, axB, axC = axs
    m = V["mech"]
    for i, v in enumerate(MAINF):
        axA.plot(i - 0.12, m[v]["Tc"], "o", mfc="white", mec=COL[v], ms=4.5, mew=1.1)
        axA.plot(i + 0.12, m[v]["Td"], "s", color=COL[v], ms=4.5)
    axA.set_yscale("log"); axA.set_ylabel("Mean episode length (proposals)")
    axA.set_yticks([100, 300, 1000, 3000]); axA.set_yticklabels(["100", "300", "1k", "3k"]); axA.yaxis.set_minor_formatter(NullFormatter())
    axA.set_xticks(range(6)); axA.set_xticklabels([LAB[v].replace(" net", "\nnet") for v in MAINF], rotation=40, ha="right", fontsize=6)
    axA.legend([Line2D([], [], marker="o", mfc="white", mec="#555555", ls=""), Line2D([], [], marker="s", color="#555555", ls="")],
               ["cooperative", "defective"], frameon=False, fontsize=6, loc="upper left")
    axA.set_title("Defective episodes are long", loc="left", fontsize=7.5)
    panel_letter(axA, "A", dx=-0.3, case="upper")
    for i, v in enumerate(MAINF):
        ec, ed = m[v]["alld_exit_coop"], m[v]["alld_exit_def"]
        if ec > 0:
            axB.plot(i - 0.12, ec, "o", color=COL[v], ms=4.5)
        else:
            axB.plot(i - 0.12, 2e-6, "v", color=COL[v], ms=4.5, mfc="white")
        axB.plot(i + 0.12, ed, "o", mfc="white", mec=COL[v], ms=4.5, mew=1.1)
    axB.set_yscale("log"); axB.set_ylim(1e-6, 0.05); axB.set_ylabel("Exit rate from ALLD (per proposal)")
    axB.set_xticks(range(6)); axB.set_xticklabels([LAB[v].replace(" net", "\nnet") for v in MAINF], rotation=40, ha="right", fontsize=6)
    axB.legend([Line2D([], [], marker="o", color="#555555", ls=""), Line2D([], [], marker="o", mfc="white", mec="#555555", ls=""),
                Line2D([], [], marker="v", mfc="white", mec="#555555", ls="")], ["to cooperative", "to other defective", "none possible"],
               frameon=False, fontsize=6, loc="lower right")
    axB.set_title("Defectors are rarely replaced\nby cooperators", loc="left", fontsize=7.5)
    panel_letter(axB, "B", dx=-0.3, case="upper")
    mh = V.get("mh", {})
    xs = {"s11": 0, "big1": 1, "big": 2}
    for var in ["CIRC16", "PROG16"]:
        for b, off, mk in [(0.0, -0.1, "s"), (1.0, 0.1, "o")]:
            for fl, dx in [(0.05, -0.04), (0.5, 0.0), (2.0, 0.04)]:
                for lab, x in xs.items():
                    r = mh.get("%s|%s|%s|%s" % (var, lab, fl, b))
                    if r is None:
                        continue
                    axC.errorbar(x + off + dx, r[0], yerr=r[1], fmt=mk, ms=3.2 if fl == 0.5 else 2.2, color=COL[var],
                                 mfc=COL[var] if b == 1.0 else "white", lw=0.6, capsize=1)
            axC.axhline(V["base"][var][0], color=COL[var], ls=":", lw=0.7)
    axC.axhline(0.5, color="#555555", ls="--", lw=0.6); axC.axhline(V["base"]["TAB2"][0], color=COL["TAB2"], ls=":", lw=0.7)
    for x, y, t, c in [(0.5, V["base"]["TAB2"][0], "Table-2", COL["TAB2"]),
                       (1.10, 0.5, "all memory-two equally likely", "#555555"),
                       (0.5, V["base"]["PROG16"][0], "program, not reweighted", COL["PROG16"]),
                       (0.5, V["base"]["CIRC16"][0], "circuit, not reweighted", COL["CIRC16"])]:
        axC.text(x, y + 0.006, t, color=c, fontsize=5.5, ha="center", va="bottom")
    axC.set_xticks(range(3)); axC.set_xticklabels(["$10^7$", "$10^8$", "$10^9$"]); axC.set_xlabel("Genotypes used to estimate the prior")
    axC.set_ylabel("Long-run cooperation"); axC.set_ylim(0.05, 0.72); axC.set_xlim(-0.4, 2.4)
    axC.legend([Line2D([], [], color=COL["CIRC16"], marker="o", ls=""), Line2D([], [], color=COL["PROG16"], marker="o", ls=""),
                Line2D([], [], color="#555555", marker="s", mfc="white", ls=""), Line2D([], [], color="#555555", marker="o", ls="")],
               ["circuit", "program", r"$\beta=0$", r"$\beta=1$"], frameon=False, fontsize=6, loc="center right", ncol=2, bbox_to_anchor=(1.0, 0.225))
    axC.set_title("Reweighting is partial and estimator-dependent", loc="left", fontsize=7.5)
    panel_letter(axC, "C", dx=-0.2, case="upper")
    _save(fig, out, "Fig5")


def figS1(V, D, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(wspace=0.4))
    axA, axB, axC = axs
    fl = V["floor_v1"]["floor_all"]; ck = {"AUTO3": "#7fcf9f", "AUTO4": "#2e9e5b", "AUTO6": "#145c33"}
    for v, k in [("AUTO3", 3), ("AUTO4", 4), ("AUTO6", 6)]:
        ms_ = sorted(int(m) for m in fl[v]); fx = [fl[v][str(m)]["floor"] for m in ms_]
        axA.plot(ms_, np.log10(fx), "--", color=ck[v], lw=0.9)
        obs = [(m, fl[v][str(m)]["mass"] / fl[v][str(m)]["n_observed"]) for m in ms_
               if fl[v][str(m)]["expected_count_at_floor"] >= 3 and fl[v][str(m)]["n_observed"] > 0]
        axA.plot([o[0] for o in obs], np.log10([o[1] for o in obs]), "o", color=ck[v], ms=3.5, label="$k=%d$" % k)
    axA.set_xlabel("States $m$"); axA.set_ylabel(r"$\log_{10}$ probability per phenotype"); axA.legend(frameon=False, fontsize=6)
    panel_letter(axA, "A", dx=-0.3, case="upper")
    order = ["TAB1", "TAB2", "CIRC8", "CIRC16", "CIRC32", "PROG8", "PROG16", "PROG32", "AUTO3", "AUTO4", "AUTO6", "RNN2", "RNN3", "RNN4"]
    fam = {"TAB1": "TAB1", "TAB2": "TAB2", "CIRC": "CIRC16", "PROG": "PROG16", "AUTO": "AUTO4", "RNN": "RNN3"}
    for i, v in enumerate(order):
        c = COL[fam[[k for k in fam if v.startswith(k)][0]]]
        m, h = V["base"][v]
        axB.errorbar(i, m, yerr=h, fmt="o", color=c, ms=3.5, capsize=1.2, lw=0.7)
    axB.set_xticks(range(len(order))); axB.set_xticklabels(["T1", "T2", "8", "16", "32", "8", "16", "32", "3", "4", "6", "2", "3", "4"], fontsize=6)
    axB.set_ylabel("Long-run cooperation"); axB.set_ylim(0, 0.65)
    for x, t in [(3, "circ."), (6, "prog."), (9, "auto."), (12, "RNN")]:
        axB.text(x, 0.61, t, ha="center", fontsize=5.5)
    panel_letter(axB, "B", dx=-0.25, case="upper")
    for v in ["TAB2", "CIRC16", "PROG16"]:
        ib = V["ibm"][v]; mus = sorted(ib, key=float)
        axC.plot([float(m) * 50 for m in mus], [ib[m] for m in mus], "-o", color=COL[v], ms=3, lw=1.0, label=LAB[v])
        axC.axhline(V["base"][v][0], color=COL[v], ls=":", lw=0.7)
    axC.set_xscale("log"); axC.set_xlabel(r"$\mu N$"); axC.set_ylabel("Long-run cooperation"); axC.set_ylim(0, 0.7)
    axC.legend(frameon=False, fontsize=6, loc="center right")
    panel_letter(axC, "C", dx=-0.25, case="upper")
    _save(fig, out, "FigS1")


def figS2(V, D, out):
    with (D["workspace"] / "res3" / "human2.json").open(encoding="utf-8") as stream:
        H = json.load(stream)
    cnt = np.array(H["counts"]); u = np.array(H["u"]); codes = H["codes"]
    share = cnt.sum(0) / cnt.sum(); upool = (u * cnt.sum(1)[:, None]).sum(0) / cnt.sum()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(wspace=0.35))
    mf = np.array(H["m_full"]); mp = np.array(H["m_path"])
    for i in range(32):
        mk = "o" if mf[i] == 1 else ("^" if mp[i] == 1 else ".")
        axA.plot(upool[i], share[i] + 1e-3, mk, color="#1f6fb4" if mf[i] == 1 else ("#e07b1a" if mp[i] == 1 else "#777777"),
                 ms=5 if mk != "." else 6)
    label_offsets = {"TFT": (-4, 3), "Grim-1": (-4, -9)}
    for c, lab in [("DDDDD", "ALLD"), ("CCCDD", "TFT"), ("CCDDD", "Grim-1"), ("DCCDD", "STFT"), ("CCCCC", "ALLC")]:
        i = codes.index(c)
        offset = label_offsets.get(lab, (4, 2))
        alignment = "right" if lab in label_offsets else "left"
        axA.annotate(lab, (upool[i], share[i] + 1e-3), xytext=offset, textcoords="offset points", fontsize=6, ha=alignment)
    axA.set_yscale("log"); axA.set_xlabel("Expected payoff against population / $(T-S)$"); axA.set_ylabel("Share of choices (+0.001)")
    panel_letter(axA, "A", dx=-0.25, case="upper")
    lz = H["lz_orderings_two_state"]; g = np.array([r["gamma_per_LZ"] for r in lz]); o = np.argsort(g)
    axB.bar(range(24), g[o], color=["#1f6fb4" if k == 0 else "#bbbbbb" for k in o])
    axB.axhline(0, color="black", lw=0.6); axB.set_xticks([]); axB.set_xlabel("Outcome ordering (sorted)")
    axB.set_ylabel(r"$\gamma$ per LZ unit (two-state strategies)")
    i0 = list(o).index(0); axB.annotate("first analysis", (i0, g[0]), xytext=(i0 - 8, g[0] + 0.08), fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
    panel_letter(axB, "B", dx=-0.25, case="upper")
    _save(fig, out, "FigS2")


FIGURES = {f.__name__.replace("fig", "Fig", 1): f for f in (fig1, fig2, fig3, fig4, fig5, figS1, figS2)}


def make_figures(workspace, output, names):
    apply_figure_style(sizes=(8, 7, 6))
    numbers, data = _load(Path(workspace))
    for name in names:
        FIGURES[name](numbers, data, output)
