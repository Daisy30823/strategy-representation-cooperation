"""analysis2.py -- builds every number of the revised manuscript, SI and response letter.
Inputs : runA/ runB/ runC/ runD/ (task outputs), priors/priors_summary.json, exact1/exact1_*.json, res3/human2.json,
         v1 archived outputs (v1/.../chains_all.csv, numbers.json) for the beta sweep and one-at-a-time settings.
Output : res3/numbers2.json  (raw values)  and  res3/plot_data.json (unrounded values for Figs. 3B and 4)
Usage  : python analysis2.py <workspace>
"""
import os, sys, json, glob
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_core2 as C

ROOT = os.path.abspath(sys.argv[1])
J = lambda *p: os.path.join(ROOT, *p)
MAIN = ["TAB1", "TAB2", "CIRC16", "PROG16", "AUTO4", "RNN3"]
ALL14 = ["TAB1", "TAB2", "CIRC8", "CIRC16", "CIRC32", "PROG8", "PROG16", "PROG32", "AUTO3", "AUTO4", "AUTO6", "RNN2", "RNN3", "RNN4"]
FAM = {"TAB1": "Table-1", "TAB2": "Table-2", "CIRC8": "Circuit", "CIRC16": "Circuit", "CIRC32": "Circuit", "PROG8": "Program",
       "PROG16": "Program", "PROG32": "Program", "AUTO3": "Automaton", "AUTO4": "Automaton", "AUTO6": "Automaton",
       "RNN2": "RNN", "RNN3": "RNN", "RNN4": "RNN"}
NM = C.named_machines()


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if f.endswith("tasks.json"):
            continue
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()) if n else np.nan, np.nan
    return float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def split_rhat(chains):
    """classical split-R-hat on window means (each chain split in two halves)."""
    S = []
    for c in chains:
        c = np.asarray(c, float); h = len(c) // 2
        S += [c[:h], c[h:2 * h]]
    S = np.array(S); n = S.shape[1]
    W = S.var(axis=1, ddof=1).mean(); B = n * S.mean(axis=1).var(ddof=1)
    var = (n - 1) / n * W + B / n
    return float(np.sqrt(var / W))


def batch_mcse(w):
    w = np.asarray(w, float)
    return float(w.std(ddof=1) / np.sqrt(len(w)))


V = {}
RA, RB, RC, RD = load_dir(J("runA")), load_dir(J("runB")), load_dir(J("runC")), load_dir(J("runD"))
R_all = {**RA, **RB, **RC, **RD}

# ------------------------------------------------------------------ chains table
rows = []
for k, d in R_all.items():
    t = d["task"]
    if t["kind"] != "chain2":
        continue
    p = t.get("prm", {})
    rows.append(dict(id=t["id"], grp=t["grp"], var=t["var"], beta=p.get("beta", 1.0), delta=p.get("delta", 0.9),
                     b=p.get("pay", [2, -1, 3, 0])[2], mut=t.get("mut", "point"), init=d["init"], coop=d["coop"],
                     ovf=d["overflow"], coll=d["collisions"], trunc=d["log_truncated"], sec=d["seconds"], key=k,
                     mh=(t.get("mh") or {}).get("file", ""), floor=(t.get("mh") or {}).get("floor", np.nan)))
CH = pd.DataFrame(rows)
V["integrity"] = dict(n_chains=int(len(CH)), overflow=int(CH.ovf.sum()), collisions=int(CH.coll.sum()),
                      truncated=int(CH.trunc.sum()), core_hours=float(sum(d.get("seconds", 0) for d in R_all.values()) / 3600))

# ------------------------------------------------------------------ v1 archive (beta sweep, one-at-a-time settings, size variants)
v1c = pd.read_csv(glob.glob(J("v1", "**", "chains_all.csv"), recursive=True)[0])
v1n = json.load(open(glob.glob(J("v1", "**", "numbers.json"), recursive=True)[0]))
P1 = v1c.groupby(["cond", "var"]).coop.mean().unstack(0)
CI1 = v1c.groupby(["cond", "var"]).coop.apply(lambda x: ci(x)[1]).unstack(0)
V["v1"] = dict(P=P1.round(6).to_dict(), CI=CI1.round(6).to_dict())

# baseline: re-run main chains (identical seeds) + archived size variants
base = {}
for v in ALL14:
    x = CH[(CH.grp == "base2") & (CH["var"] == v)].coop.values
    if len(x) == 0:
        x = v1c[(v1c.cond == "base") & (v1c["var"] == v)].coop.values
    base[v] = ci(x)
V["base"] = base
v1base_main = {v: float(v1c[(v1c.cond == "base") & (v1c["var"] == v)].coop.mean()) for v in MAIN}
V["base_rerun_maxdiff"] = float(max(abs(base[v][0] - v1base_main[v]) for v in MAIN))

# ------------------------------------------------------------------ convergence diagnostics
conv = {}
for v in MAIN:
    L_ = CH[(CH.grp == "long") & (CH["var"] == v)]
    if len(L_) == 0:
        continue
    W = [np.array(R_all[k]["windows"]) for k in L_.key]
    conv[v] = dict(rhat=split_rhat(W), mean=float(np.mean([w.mean() for w in W])),
                   by_init={i: float(L_[L_.init == i].coop.mean()) for i in L_.init.unique()},
                   mcse=float(np.sqrt(np.mean([batch_mcse(w) ** 2 for w in W]) / len(W))),
                   base_mean=base[v][0], diff_long_base=float(L_.coop.mean() - base[v][0]))
    B_ = CH[(CH.grp == "base2") & (CH["var"] == v)]
    Wb = [np.array(R_all[k]["windows"]) for k in B_.key]
    conv[v]["rhat_base"] = split_rhat(Wb)
V["conv"] = conv

# ------------------------------------------------------------------ crossed game grid
G = CH[CH.grp == "grid"].groupby(["delta", "b", "var"]).coop.mean().unstack("var")
for v in MAIN:
    G.loc[(0.9, 3.0), v] = base[v][0]
G = G.sort_index()[MAIN]
Y = G.values; gm = Y.mean()
SSt = ((Y - gm) ** 2).sum(); SSe = Y.shape[0] * ((Y.mean(0) - gm) ** 2).sum(); SSs = Y.shape[1] * ((Y.mean(1) - gm) ** 2).sum()
V["grid"] = dict(table=G.round(4).reset_index().to_dict(orient="records"), share_enc=float(SSe / SSt), share_set=float(SSs / SSt),
                 share_int=float(1 - SSe / SSt - SSs / SSt), n_cells=int(Y.size))
# decomposition restricted to settings above the reciprocity threshold delta > c/b with b-1 = cost 1
coop_reg = [(d, b) for d, b in G.index if d > 1.0 / b + 0.2]
Yr = G.loc[coop_reg].values; gmr = Yr.mean(); SSt2 = ((Yr - gmr) ** 2).sum()
V["grid"]["regime_settings"] = [list(x) for x in coop_reg]
V["grid"]["share_enc_coopregime"] = float(Yr.shape[0] * ((Yr.mean(0) - gmr) ** 2).sum() / SSt2)
V["grid"]["share_set_coopregime"] = float(Yr.shape[1] * ((Yr.mean(1) - gmr) ** 2).sum() / SSt2)
V["grid"]["spread_by_setting"] = {f"{d}_{b}": float(G.loc[(d, b)].max() - G.loc[(d, b)].min()) for d, b in G.index}
V["grid"]["delta_effect_b3"] = {v: float(G.loc[(0.9, 3.0), v] - G.loc[(0.7, 3.0), v]) for v in MAIN}
V["grid"]["low_delta_max"] = float(G.loc[[(0.3, 1.5), (0.3, 3.0), (0.5, 1.5)]].values.max())
# one-at-a-time (archived) decomposition, kept for comparison with the first analysis
cols12 = ["base", "eps0", "eps0.05", "delta0.8", "delta0.95", "moran", "N20", "N100", "pdT5", "beta0.1", "beta0.3", "beta3"]
Y12 = P1.loc[MAIN, cols12].values; g12 = Y12.mean(); S12 = ((Y12 - g12) ** 2).sum()
V["ofat"] = dict(share_enc=float(Y12.shape[1] * ((Y12.mean(1) - g12) ** 2).sum() / S12),
                 share_set=float(Y12.shape[0] * ((Y12.mean(0) - g12) ** 2).sum() / S12))
rk = pd.DataFrame(Y12, index=MAIN).rank(ascending=False)
V["ofat"]["circuit_lowest_all"] = bool((pd.DataFrame(Y12, index=MAIN).idxmin() == "CIRC16").all())
V["ofat"]["n_rank_patterns"] = int(len(set(tuple(rk[c].values) for c in rk.columns)))

# ------------------------------------------------------------------ house-of-cards and valence controls, IBM
H = CH[CH.grp == "hoc"].groupby(["var", "beta"]).coop.mean().unstack("beta")
V["hoc"] = H.round(4).to_dict(orient="index")
V["hoc_ci"] = CH[CH.grp == "hoc"].groupby(["var", "beta"]).coop.apply(lambda x: ci(x)[1]).unstack("beta").round(4).to_dict(orient="index")
VAL = CH[CH.grp == "valence"].groupby(["var", "beta"]).coop.apply(lambda x: ci(x)).unstack("beta")
V["valence"] = {v: {str(b): list(VAL.loc[v, b]) for b in VAL.columns} for v in VAL.index}
V["valence_prior"] = {}
for v in ["CIRC16C", "PROG16C", "PROG16dC", "RNN3C"]:
    d = R_all.get("sample2_samp__" + v)
    if d:
        V["valence_prior"][v] = dict(prior=d["prior_coop"], M=d["M"], named={k: c / d["M"] for k, c in d["named_counts"].items()},
                                     mass_by_n=d["mass_by_n"])
ib = pd.DataFrame([dict(var=d["task"]["var"], mu=d["task"]["mu"], coop=d["coop"], mono=d.get("mono")) for d in R_all.values() if d["task"]["kind"] == "ibm"])
V["ibm"] = {v: {str(m): float(ib[(ib["var"] == v) & (ib.mu == m)].coop.mean()) for m in sorted(ib.mu.unique())} for v in ib["var"].unique()}
V["ibm_mono"] = {v: {str(m): float(ib[(ib["var"] == v) & (ib.mu == m)].mono.mean()) for m in sorted(ib.mu.unique())} for v in ib["var"].unique()}

# ------------------------------------------------------------------ verification chains vs exact
EX = {os.path.basename(f)[7:-5]: json.load(open(f)) for f in sorted(glob.glob(J("exact1", "exact1_*.json")))}
ver = {}
for v in EX:
    e = EX[v]["dyn"]
    for b in [0.1, 1.0]:
        x = CH[(CH.grp == "verif") & (CH["var"] == v) & (CH.beta == b) & (CH.mut == "point")].coop.values
        if len(x):
            ver[f"{v}_b{b}"] = dict(exact=e[f"point_b{b:g}"]["coop"], chain=ci(x))
    x = CH[(CH.grp == "verif") & (CH["var"] == v) & (CH.mut == "hoc")].coop.values
    if len(x):
        ver[f"{v}_hoc_b1.0"] = dict(exact=e["hoc_b1"]["coop"], chain=ci(x))
V["verif"] = ver
if ver:
    V["verif_maxabs"] = float(max(abs(r["exact"] - r["chain"][0]) for r in ver.values()))
    V["verif_inside"] = int(sum(abs(r["exact"] - r["chain"][0]) <= 1.5 * r["chain"][1] for r in ver.values()))
    V["verif_n"] = len(ver)
V["exact"] = {v: dict(n_geno=e["n_geno"], n_pheno=e["n_pheno"], prior=e["prior_coop"], uniform=e["uniform_coop"],
                      dyn={k: x["coop"] for k, x in e["dyn"].items()},
                      p_alld=sum(p["prior"] for p in e["phen"] if p["label"] == "ALLD"),
                      p_tft=sum(p["prior"] for p in e["phen"] if p["label"] == "TFT"),
                      p_stft=sum(p["prior"] for p in e["phen"] if p["label"] == "STFT"),
                      p_one=sum(p["prior"] for p in e["phen"] if p["n"] == 1)) for v, e in EX.items()}

# ------------------------------------------------------------------ sampling (genotype-phenotype maps)
samp = {}
for tag, M_ in [("samp3e6", 3e6), ("samp1e7", 1e7)]:
    for v in ALL14:
        f = J("runB", "phen_%s__%s.npz" % (tag, v))
        if not os.path.exists(f):
            continue
        z = np.load(f); cnt = z["cnt"].astype(float); n = z["n"]; coop = z["coop"]; klz = z["klz"]; M = float(z["M"])
        p = cnt / M; lp = np.log(p)
        d = R_all["sample2_%s__%s" % (tag, v)]
        r = dict(M=int(M), distinct=int(len(cnt)), singletons=int((cnt == 1).sum()), gt_unseen=float((cnt == 1).sum() / M),
                 H=float(-(p * np.log2(p)).sum()), Hmax=float(np.log2(M)), prior=float((p * coop).sum()),
                 mass_by_n={int(k): float(p[n == k].sum()) for k in np.unique(n)},
                 named={k: c / M for k, c in d["named_counts"].items()})
        r["mass_le2"] = r["mass_by_n"].get(1, 0) + r["mass_by_n"].get(2, 0)
        if v not in ("TAB1", "TAB2"):
            r["rho_m_all"] = float(spearmanr(lp, n)[0]); r["rho_K_all"] = float(spearmanr(lp, klz)[0])
            m5 = cnt >= 5
            r["rho_K_5"] = float(spearmanr(lp[m5], klz[m5])[0]) if m5.sum() > 10 else np.nan
            r["rho_m_5"] = float(spearmanr(lp[m5], n[m5])[0]) if m5.sum() > 10 else np.nan
            r["n5"] = int(m5.sum())
            # weighted (by probability) mean complexity vs uniform reference
            r["mean_m"] = float((p * n).sum()); r["mean_K"] = float((p * klz).sum())
            # envelope: max log10 P per integer-rounded K
            Kr = np.round(klz).astype(int); env = {}
            for kk in np.unique(Kr):
                env[int(kk)] = float(np.log10(p[Kr == kk].max()))
            r["envelope"] = env
        samp[(tag, v)] = r
V["samp"] = {f"{a}|{b}": r for (a, b), r in samp.items()}
# exact references for the tables
ns2, hs2 = C.enumerate_table2(C.FEAT)
mass2 = {int(k): float((ns2 == k).mean()) for k in np.unique(ns2)}
V["tab2_mass_by_n"] = mass2
V["tab2_le2"] = mass2.get(1, 0) + mass2.get(2, 0)
V["tab2_n_le2"] = int((ns2 <= 2).sum())
V["tab2_prior"] = float(v1n["prior"]["TAB2"])
V["tab1_exact"] = dict(prior=EX["TAB1"]["prior_coop"] if "TAB1" in EX else None,
                       mass_by_n={int(k): float(sum(p["prior"] for p in EX["TAB1"]["phen"] if p["n"] == k)) for k in range(1, 6)} if "TAB1" in EX else {})
# enrichment vs Table-2 for circuit and program
enr = {}
for v in ["CIRC16", "PROG16"]:
    r = samp.get(("samp1e7", v))
    if r:
        enr[v] = dict(one=r["mass_by_n"].get(1, 0) / mass2[1], le2=r["mass_le2"] / V["tab2_le2"],
                      n9=r["mass_by_n"].get(9, 0) / mass2.get(9, np.nan))
V["enrich"] = enr
# family-level association between arrival statistics and long-run cooperation
fam_rows = []
for v in ALL14:
    r = samp.get(("samp3e6", v))
    if r is None:
        continue
    fam_rows.append(dict(var=v, fam=FAM[v], coop=base[v][0], palld=r["named"]["ALLD"] if v != "TAB2" else 2 ** -21,
                         p1=r["mass_by_n"].get(1, 0), prior=r["prior"]))
FR = pd.DataFrame(fam_rows)
if len(FR):
    main_rows = FR[FR["var"].isin(MAIN)]
    V["assoc"] = dict(rho14_alld=float(spearmanr(FR.palld, FR.coop)[0]), rho14_p1=float(spearmanr(FR.p1, FR.coop)[0]),
                      rho6_alld=float(spearmanr(main_rows.palld, main_rows.coop)[0]), rho6_p1=float(spearmanr(main_rows.p1, main_rows.coop)[0]),
                      rho6_prior=float(spearmanr(main_rows.prior, main_rows.coop)[0]),
                      fammean=FR.groupby("fam")[["coop", "palld", "p1"]].mean().to_dict(orient="index"))

# ------------------------------------------------------------------ mechanism (baseline re-runs, uncapped)
mech = {}
for v in MAIN:
    ks = list(CH[(CH.grp == "base2") & (CH["var"] == v)].key)
    if not ks:
        continue
    D = [R_all[k] for k in ks]
    tot = np.array([d["n_prop"] - d["task"]["burn"] for d in D], float)
    occD = np.array([d["occ"]["ALLD"] for d in D]); occC = np.array([d["occ"]["ALLC"] for d in D])
    m = dict(
        alld_share=float(np.mean(occD[:, 0] / tot)), allc_share=float(np.mean(occC[:, 0] / tot)),
        coop_time=float(np.mean([d["frac_time_coop"] for d in D])),
        Tc=float(np.mean([d["ep_coop"]["mean"] for d in D])), Td=float(np.mean([d["ep_def"]["mean"] for d in D])),
        Tc_med=float(np.mean([d["ep_coop"]["median"] for d in D])), Td_med=float(np.mean([d["ep_def"]["median"] for d in D])),
        n_down=float(np.mean([d["n_down"] for d in D])),
        alld_soj=float(np.mean([d["alld_sojourn"]["mean"] for d in D if d["alld_sojourn"]["mean"]])),
        alld_exit_coop=float(occD[:, 3].sum() / occD[:, 0].sum()), alld_exit_def=float(occD[:, 4].sum() / occD[:, 0].sum()),
        alld_neutral_moves=float(occD[:, 2].sum() / occD[:, 0].sum()),
        allc_exit_def=float(occC[:, 4].sum() / max(occC[:, 0].sum(), 1)), allc_exit_coop=float(occC[:, 3].sum() / max(occC[:, 0].sum(), 1)),
    )
    for f_ in ["frac_pre_vulnerable", "frac_founder_robust", "frac_drift", "frac_drift_given_robust"]:
        m[f_] = float(np.mean([d["collapse"].get(f_, np.nan) for d in D]))
    # neighbourhoods of recorded residents (occupancy-weighted)
    def nbm(lab):
        X = [np.array(d[lab]["mean"]) * d[lab]["n"] for d in D if d.get(lab)]
        n_ = sum(d[lab]["n"] for d in D if d.get(lab))
        return (np.sum(X, axis=0) / n_, n_) if n_ else (None, 0)
    xc, nc = nbm("nbhd_coop"); xd, nd = nbm("nbhd_alld")
    if xc is not None:
        nn = xc[11]
        m.update(nbc_n=nc, nbc_neutral=xc[0], nbc_spoiler=xc[1], nbc_spoiler_any=xc[2], nbc_invader=xc[3],
                 nbc_spoiler_nn=xc[1] / nn if nn > 0 else np.nan, nbc_spoiler_any_nn=xc[2] / nn if nn > 0 else np.nan,
                 nbc_invader_nn=xc[3] / nn if nn > 0 else np.nan, nbc_nearneutral_nn=xc[4] / nn if nn > 0 else np.nan,
                 nbc_V=xc[9])
    if xd is not None:
        m.update(nbd_n=nd, nbd_neutral=xd[0], nbd_rate_coop=xd[7], nbd_rate_def=xd[8], nbd_nearneutral=xd[4])
    f2 = R_all.get("focal2_focal2__" + v)
    if f2:
        for nm_ in ["ALLD", "ALLC", "TFT", "STFT", "WSLS"]:
            q = f2.get(nm_)
            if q and q["n"]:
                m["u_" + nm_] = dict(n=q["n"], neutral=q["mean"][0], rate_coop=q["mean"][7], rate_def=q["mean"][8], near=q["mean"][4])
    mech[v] = m
V["mech"] = {v: {k: (float(x) if isinstance(x, (float, np.floating, int, np.integer)) else x) for k, x in m.items()} for v, m in mech.items()}
# grid episode/ALLD share trend
# ------------------------------------------------------------------ support and priors for reweighting
pe = [R_all[k] for k in R_all if k.startswith("progenum")]
if pe:
    cnt8 = sum(np.load(J("runA", "progenum_%s.npz" % d["task"]["id"]))["cnt"].astype(np.int64) for d in pe)
    V["prog8_support"] = int((cnt8 > 0).sum()); V["prog8_total"] = int(cnt8.sum())
V["priors"] = json.load(open(J("priors", "priors_summary.json")))
MH = CH[CH.grp == "mh"].copy()
if len(MH):
    MH["lab"] = MH.mh.str.extract(r"prior_\w+?_(\w+)\.npz")[0]
    V["mh"] = {f"{v}|{lab}|{fl}|{b}": list(ci(g.coop)) for (v, lab, fl, b), g in MH.groupby(["var", "lab", "floor", "beta"])}
V["mh_v1"] = {v: float(P1.loc[v, "mh"]) for v in ["CIRC16", "PROG16", "AUTO4", "RNN3"] if v in P1.index}
V["mh_v1_beta0"] = {v: float(P1.loc[v, "mhbeta0"]) for v in ["CIRC16", "PROG16", "AUTO4", "RNN3"] if v in P1.index and not np.isnan(P1.loc[v, "mhbeta0"])}


# ------------------------------------------------------------------ priors of the main encodings, neutral-chain check, floor check
prior_main = {}
for v in MAIN:
    if v == "TAB1" and "TAB1" in EX:
        prior_main[v] = EX["TAB1"]["prior_coop"]
    elif v == "TAB2":
        prior_main[v] = float(v1n["prior"]["TAB2"])
    else:
        r = samp.get(("samp1e7", v)) or samp.get(("samp3e6", v))
        prior_main[v] = r["prior"] if r else float(v1n["prior"][v])
V["prior_main"] = prior_main
V["beta0_maxdiff"] = float(max(abs(P1.loc[v, "beta0"] - prior_main[v]) for v in MAIN))
fa = v1n["floor"]["AUTO4"]["2"]
V["floor_v1"] = dict(min_count_A4_m2=int(fa["min_count"]), exp_A4_m2="%.0f" % fa["expected_count_at_floor"], floor_all=v1n["floor"])
import numba, scipy, platform
V["genome"] = {v: (int(len(C.gene_alphabet2(*{'TAB1':(0,0),'TAB2':(1,0),'CIRC8':(2,8),'CIRC16':(2,16),'CIRC32':(2,32),'PROG8':(3,8),'PROG16':(3,16),'PROG32':(3,32),'AUTO3':(4,3),'AUTO4':(4,4),'AUTO6':(4,6),'RNN2':(5,2),'RNN3':(5,3),'RNN4':(5,4)}[v]))), float(np.log2(C.gene_alphabet2(*{'TAB1':(0,0),'TAB2':(1,0),'CIRC8':(2,8),'CIRC16':(2,16),'CIRC32':(2,32),'PROG8':(3,8),'PROG16':(3,16),'PROG32':(3,32),'AUTO3':(4,3),'AUTO4':(4,4),'AUTO6':(4,6),'RNN2':(5,2),'RNN3':(5,3),'RNN4':(5,4)}[v]).astype(float)).sum())) for v in ALL14}
V["versions"] = dict(pyver=platform.python_version(), npver=np.__version__, spver=scipy.__version__, nbver=numba.__version__)

# ------------------------------------------------------------------ human data
V["human"] = json.load(open(J("res3", "human2.json")))
for k in ["codes", "counts", "u", "klz_identity", "klz_mean", "m_full", "m_path"]:
    V["human"].pop(k, None)

json.dump(V, open(J("res3", "numbers2.json"), "w"), indent=1, default=float)
with open(J("res3", "plot_data.json"), "w", encoding="utf-8") as stream:
    json.dump({"P1": P1.to_dict(), "G": G.reset_index().to_dict(orient="records")}, stream)
print("ok", len(CH), "chains")
