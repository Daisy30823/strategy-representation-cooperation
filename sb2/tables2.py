"""Render the published table rows using the original Dataset S1 formatters."""
import json

import numpy as np

MAIN = ["TAB1", "TAB2", "CIRC16", "PROG16", "AUTO4", "RNN3"]
ALL14 = ["TAB1", "TAB2", "CIRC8", "CIRC16", "CIRC32", "PROG8", "PROG16", "PROG32", "AUTO3", "AUTO4", "AUTO6", "RNN2", "RNN3", "RNN4"]
NAME = {"TAB1": "Table-1", "TAB2": "Table-2", "CIRC16": "Circuit", "PROG16": "Program", "AUTO4": "Automaton", "RNN3": "Recurrent net",
        "CIRC8": "Circuit, $G=8$", "CIRC32": "Circuit, $G=32$", "PROG8": "Program, $L=8$", "PROG32": "Program, $L=32$",
        "AUTO3": "Automaton, $k=3$", "AUTO6": "Automaton, $k=6$", "RNN2": "Recurrent net, $H=2$", "RNN4": "Recurrent net, $H=4$"}
NAME14 = dict(NAME, CIRC16="Circuit, $G=16$", PROG16="Program, $L=16$", AUTO4="Automaton, $k=4$", RNN3="Recurrent net, $H=3$")
EXN = {"TAB1": "memory-one table", "CIRC1": "two-gate circuit", "PROG1": "four-instruction program", "PERC1": "perceptron",
       "RNN1t": "one-unit network", "AUTO2": "two-state automaton"}


def _neg(s):
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def f3(x):
    return _neg("%.3f" % x)


def f2(x):
    return _neg("%.2f" % x)


def sci(x, d=1):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return ("%." + str(d) + "f\\times10^{%d}") % (m, e)


def big(n):
    return "{:,}".format(int(n)).replace(",", "{,}")


def bci(V, v):
    m, h = V["base"][v]
    return "%.3f $\\pm$ %.3f" % (m, h)


def samp(V, v, tag="samp1e7"):
    s = V["samp"].get(tag + "|" + v)
    if s is None and tag == "samp1e7":
        s = V["samp"].get("samp3e6|" + v)
    return s


GENES = {"TAB1": "actions", "TAB2": "actions", "CIRC16": "gate types, inputs", "PROG16": "instructions", "AUTO4": "outputs, transitions",
         "RNN3": "weights, initial state"}
SPACE = {"TAB1": "memory-one (32)", "TAB2": "memory-two ($2^{21}$)", "CIRC16": "memory-two ($2^{21}$)", "PROG16": "memory-two ($2^{21}$)",
         "AUTO4": "machines, $\\le4$ states", "RNN3": "machines, $\\le8$ states"}


def pnum(x):
    if x is None or x != x:
        return "--"
    if x == 0:
        return "0"
    return f3(x) if x >= 1e-3 else "$" + sci(x) + "$"


def table1_rows(V):
    rows = []
    for v in MAIN:
        Lg, lg = V["genome"][v]
        if v == "TAB1":
            dist, pa, pt, ps = "32", 1 / 32, 1 / 32, 1 / 32
            le2 = sum(V["tab1_exact"]["mass_by_n"].get(str(k), 0) for k in (1, 2))
        elif v == "TAB2":
            dist, pa, pt, ps, le2 = big(2 ** 21), 2 ** -21, 2 ** -21, 2 ** -21, V["tab2_le2"]
        else:
            s = samp(V, v); dist = big(s["distinct"]); pa, pt, ps, le2 = s["named"]["ALLD"], s["named"]["TFT"], s["named"]["STFT"], s["mass_le2"]
        rows.append("%s & %s & %s & %d & %.1f & %s & %s & %s & %s & %s & %s & %s\\\\" % (
            NAME[v], SPACE[v], GENES[v], Lg, lg, dist, pnum(pa), pnum(pt), pnum(ps), pnum(le2), f3(V["prior_main"][v]), bci(V, v)))
    return "\n".join(rows)


def tS1(V):
    rows = []
    for v in ALL14:
        s = samp(V, v, "samp3e6")
        if s is None:
            continue
        Lg, lg = V["genome"][v]
        rho = lambda k: (f2(s[k]) if k in s and s[k] == s[k] else "--")
        b = V["base"][v]
        if v == "TAB2":
            q = 2.0 ** -21
            rows.append("%s & %d & %.1f & %s & -- & 0 & 21.0 & %s & %s & %s & %s & %s & -- & -- & -- & %s & %s\\\\" % (
                NAME14[v], Lg, lg, big(2 ** 21) + "$^*$", pnum(q), pnum(q), pnum(q), pnum(q), pnum(V["tab2_le2"]), f3(V["prior_main"]["TAB2"]),
                "%.3f$\\pm$%.3f" % (b[0], b[1])))
            continue
        if v == "TAB1":
            q = 1 / 32
            rows.append("%s & %d & %.1f & %s & -- & 0 & 5.0 & %s & %s & %s & %s & 1.000 & -- & -- & -- & %s & %s\\\\" % (
                NAME14[v], Lg, lg, "32$^*$", pnum(q), pnum(q), pnum(q), pnum(q), f3(V["prior_main"]["TAB1"]), "%.3f$\\pm$%.3f" % (b[0], b[1])))
            continue
        rows.append("%s & %d & %.1f & %s & %s & %s & %.1f & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (
            NAME14[v], Lg, lg, big(s["distinct"]), f2(s["singletons"] / s["distinct"]), pnum(s["gt_unseen"]), s["H"],
            pnum(s["named"]["ALLD"]), pnum(s["named"]["ALLC"]), pnum(s["named"]["TFT"]), pnum(s["named"]["STFT"]),
            pnum(s["mass_le2"]), rho("rho_m_all"), rho("rho_K_all"), rho("rho_K_5"), f3(s["prior"]), "%.3f$\\pm$%.3f" % (b[0], b[1])))
    return "\n".join(rows)


def tS2(V):
    P1, CI1 = V["v1"]["P"], V["v1"]["CI"]
    lab = [("base", "Baseline"), ("eps0", "$\\varepsilon=0$"), ("eps0.05", "$\\varepsilon=0.05$"), ("delta0.8", "$\\delta=0.8$"),
           ("delta0.95", "$\\delta=0.95$"), ("N20", "$N=20$"), ("N100", "$N=100$"), ("moran", "Moran, $w=0.5$"),
           ("pdT5", "PD $3,0,5,1$"), ("beta0", "$\\beta=0$"), ("beta0.01", "$\\beta=0.01$"), ("beta0.1", "$\\beta=0.1$"),
           ("beta0.3", "$\\beta=0.3$"), ("beta3", "$\\beta=3$")]
    rows = []
    for c, l in lab:
        cells = []
        for v in MAIN:
            m = P1[c].get(v); h = CI1[c].get(v)
            if c == "base":
                m, h = V["base"][v]
            cells.append("--" if m is None or m != m else "%.3f$\\pm$%.3f" % (m, h))
        rows.append(l + " & " + " & ".join(cells) + "\\\\")
    return "\n".join(rows)


def tS3(V):
    m = V["mech"]; rows = []
    for v in MAIN:
        x = m[v]
        g = lambda k, d=3: ("%." + str(d) + "f") % x[k] if k in x and x[k] == x[k] else "--"
        rows.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (
            NAME[v], g("alld_share", 2), g("coop_time", 2), big(round(x["Tc"])), big(round(x["Td"])), big(round(x["n_down"])),
            g("frac_pre_vulnerable", 2), g("frac_founder_robust", 2), g("frac_drift", 2), g("frac_drift_given_robust", 2),
            g("nbc_neutral", 2), g("nbc_spoiler_any_nn") + " & " + g("nbc_spoiler_nn"), g("nbc_invader_nn"), big(round(x["alld_soj"]))))
    return "\n".join(rows)


def tS4(H):
    aka = {"CCCCC": "ALLC", "CCCCD": "AC$'$", "CCCDD": "TFT", "CCDDC": "WSLS", "CCDDD": "Grim-1", "DCCDD": "STFT", "DCDCD": "AD$'$",
           "DCDDD": "AD$'$", "DDDCD": "AD$'$", "DDDDD": "ALLD"}
    cnt = np.array(H["counts"]); rows = []
    for i, c in enumerate(H["codes"]):
        n = cnt[:, i]
        if n.sum() == 0:
            continue
        rows.append("%s & %s & %s & %d & %d & %d & %.1f & %.1f\\\\" % (c, aka.get(c, ""), " & ".join(str(int(x)) for x in n), int(n.sum()),
                                                                      int(H["m_full"][i]), int(H["m_path"][i]), H["klz_identity"][i], H["klz_mean"][i]))
    zero = [c for i, c in enumerate(H["codes"]) if cnt[:, i].sum() == 0]
    return "\n".join(rows), ", ".join(zero)


def tS5(V):
    rows = []
    for v in MAIN:
        x = V["mech"][v]
        for nm in ["ALLD", "ALLC", "TFT", "STFT", "WSLS"]:
            u = x.get("u_" + nm)
            if u:
                rows.append("%s & %s & uniform & %d & %s & %s & %s\\\\" % (NAME[v], nm, u["n"], f3(u["neutral"]), pnum(u["rate_coop"]), pnum(u["rate_def"])))
        if x.get("nbd_n"):
            rows.append("%s & ALLD & visited & %d & %s & %s & %s\\\\" % (NAME[v], x["nbd_n"], f3(x["nbd_neutral"]), pnum(x["nbd_rate_coop"]), pnum(x["nbd_rate_def"])))
        rows.append("%s & ALLD & occupancy & -- & -- & %s & %s\\\\" % (NAME[v], pnum(x["alld_exit_coop"]), pnum(x["alld_exit_def"])))
        rows.append("\\addlinespace")
    return "\n".join(rows)


def tS6(V):
    rows = []
    for k, r in V["verif"].items():
        v, rest = k.split("_", 1)
        mode = "house-of-cards" if "hoc" in rest else "point"
        b = rest.split("b")[-1]
        rows.append("%s & %s & %s & %.4f & %.4f$\\pm$%.4f\\\\" % (EXN[v], mode, b, r["exact"], r["chain"][0], r["chain"][1]))
    return "\n".join(rows)


def tS7(V):
    rows = []
    for v in MAIN:
        c = V["conv"][v]
        bi = c["by_init"]
        ib = V["ibm"].get(v, {})
        rows.append("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %s & %s & %s\\\\" % (
            NAME[v], c["base_mean"], c["mean"], bi.get("ALLD", np.nan), bi.get("TFT", np.nan), c["rhat"], c["rhat_base"],
            f3(ib["0.001"]) if ib else "--", f3(ib["0.004"]) if ib else "--", f3(ib["0.02"]) if ib else "--"))
    return "\n".join(rows)


def tS8(V):
    rows = []
    for r in V["grid"]["table"]:
        rows.append("%.1f & %.1f & %s\\\\" % (r["delta"], r["b"], " & ".join(f3(r[v]) for v in MAIN)))
    return "\n".join(rows)


def tS9(V):
    mh = V.get("mh", {}); rows = []
    for var in ["CIRC16", "PROG16"]:
        for lab, M_ in [("s11", "$10^{7}$"), ("big1", "$10^{8}$"), ("big", "$10^{9}$")]:
            for fl in [0.05, 0.5, 2.0]:
                a = mh.get("%s|%s|%s|0.0" % (var, lab, fl)); b = mh.get("%s|%s|%s|1.0" % (var, lab, fl))
                if a is None:
                    continue
                rows.append("%s & %s & %s & %.3f$\\pm$%.3f & %.3f$\\pm$%.3f\\\\" % (NAME[var], M_, fl, a[0], a[1], b[0], b[1]))
    return "\n".join(rows)


def write_tables(workspace, output):
    results = workspace / "res3"
    with (results / "numbers2.json").open(encoding="utf-8") as stream:
        numbers = json.load(stream)
    with (results / "human2.json").open(encoding="utf-8") as stream:
        human = json.load(stream)
    human_rows, zero_codes = tS4(human)
    tables = {
        "Table1": table1_rows(numbers),
        "TableS1": tS1(numbers),
        "TableS2": tS2(numbers),
        "TableS3": tS3(numbers),
        "TableS4": human_rows,
        "TableS5": tS5(numbers),
        "TableS6": tS6(numbers),
        "TableS7": tS7(numbers),
        "TableS8": tS8(numbers),
        "TableS9": tS9(numbers),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (output / (name + ".tex")).write_text(rows + "\n", encoding="utf-8")
    (output / "TableS4_zero_counts.txt").write_text(zero_codes + "\n", encoding="utf-8")
