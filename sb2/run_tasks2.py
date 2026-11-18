"""run_tasks2.py -- task runner for the revision (v2).
Usage:  python run_tasks2.py <tasks.json> <outdir> <worker_id>
Workers claim tasks through exclusive lock files in <outdir>/claims and write one JSON (or NPZ) per task.
Several workers can run concurrently on the same task list; finished tasks are skipped.
"""
import os, sys, json, time, math, traceback
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_core2 as C

NM = C.named_machines()
NAME_OF = {v[3]: k for k, v in NM.items()}
HD, HC = NM["ALLD"][3], NM["ALLC"][3]
BASE = dict(pay=[2.0, -1.0, 3.0, 0.0], delta=0.9, eps=0.01, N=50, beta=1.0, rule=0)
FEATS = {"D": C.FEAT, "C": C.FEAT_C, "M1": C.FEAT_M1}


def padm(n, o, t):
    oo = np.zeros(C.NMAX, np.int64); tt = np.zeros((C.NMAX, 4), np.int64)
    oo[:n] = o[:n]; tt[:n] = t[:n]
    return n, oo, tt


def mstr(n, o, t):
    return ";".join("%d:%s" % (int(o[s]), "".join(str(int(t[s, k])) for k in range(4))) for s in range(n))


def prm_of(task):
    prm = dict(BASE); prm.update(task.get("prm", {}))
    return prm


def vuln(n, o, t, prm):
    R, S, T, P = prm["pay"]
    nD, oD, tD = padm(*NM["ALLD"][:3])
    rAB, rBA, cB, cA = C.rho_pair(n, o, t, nD, oD, tD, prm["eps"], prm["delta"], R, S, T, P, prm["N"], prm["beta"], prm["rule"])
    return prm["N"] * rAB


def init_genotype(task, a, feat):
    enc, size, seed = task["enc"], task["size"], task["seed"]
    ini = task.get("init", "random")
    rng = np.random.default_rng(seed)
    g0 = np.array([rng.integers(x) for x in a], np.int64)
    if ini == "random":
        return g0, "random"
    G, tries = C.sample_network2(enc, size, a, NM[ini][3], 1, 50_000_000, seed + 17, feat)
    if len(G) == 0:
        return g0, "random(fallback)"
    return G[0].astype(np.int64), ini


def load_mh(task):
    if not task.get("mh"):
        return False, np.zeros(1, np.int64), np.zeros(1), 0.0
    with np.load(task["mh"]["file"]) as d:
        keys = d["keys"].astype(np.int64)
        cnt = d["cnt"].astype(np.float64)
        M = float(d["M"])
    fl = task["mh"].get("floor", 0.5)
    p = np.where(cnt > 0, cnt / M, fl / M)
    order = np.argsort(keys)
    return True, keys[order], np.log(p[order]), math.log(fl / M)


def episodes(seq_t, seq_p, sc, burn, tend):
    """episode lengths (cooperative: c>=0.5; defective: c<0.5) from the change log, excluding censored
    first/last episodes; also ALLD sojourn lengths."""
    out = {"coop": [], "def": []}
    if len(seq_t) == 0:
        return out
    coop = sc[seq_p] >= 0.5
    start = None; state = None
    for k in range(len(seq_t)):
        st = bool(coop[k])
        if state is None:
            state = st; start = None  # first episode censored
            continue
        if st != state:
            if start is not None:
                out["coop" if state else "def"].append(seq_t[k] - start)
            start = seq_t[k]; state = st
    return out


def do_chain2(task, outdir):
    prm = prm_of(task)
    enc, size, seed = task["enc"], task["size"], task["seed"]
    feat = FEATS[task.get("feat", "M1" if enc >= 8 else "D")]
    R, S, T, P = prm["pay"]
    a = C.gene_alphabet2(enc, size)
    g0, ini = init_genotype(task, a, feat)
    mh, keys, logp, floor = load_mh(task)
    cap = task.get("cap", 1500000)
    nsteps = task["nsteps"]; burn = task["burn"]; rec = task.get("rec", 20000); nwin = task.get("nwin", 40)
    mode = 1 if task.get("mut", "point") == "hoc" else 0
    t0 = time.time()
    (stats, Rn, Ro, Rt, Rh, sp, sc, tin, lt, lpid, recg, recp, win, occ) = C.run_chain2(
        enc, size, a, g0, nsteps, seed, prm["N"], prm["beta"], prm["rule"], R, S, T, P,
        prm["eps"], prm["delta"], mh, keys, logp, floor, rec, cap, feat, burn, mode, nwin, HD, HC)
    secs = time.time() - t0
    tot = stats[0]
    res = dict(task=task, prm=prm, init=ini, seconds=secs, coop=stats[1] / tot, payoff=stats[2] / tot,
               n_prop=int(stats[3]), frac_neutral_prop=stats[4] / stats[3], n_acc=int(stats[5]),
               n_acc_neutral=int(stats[6]), n_pheno=int(stats[7]), overflow=int(stats[8]),
               n_changes=int(stats[9]) - 1, collisions=int(stats[11]), wlen=int(stats[12]),
               log_truncated=int(stats[9] >= 6000000))
    res["windows"] = (win / stats[12]).tolist()
    res["occ"] = {"ALLD": occ[0].tolist(), "ALLC": occ[1].tolist()}
    share = tin / tot
    vis = np.where(tin > 0)[0]
    klz = C.complexities(Rn[vis], Ro[vis], Rt[vis], 4)
    res["mean_n_resident"] = float((share[vis] * Rn[vis]).sum())
    res["mean_klz_resident"] = float((share[vis] * klz).sum())
    res["n_hist"] = {int(k): float(share[vis][Rn[vis] == k].sum()) for k in np.unique(Rn[vis])}
    res["named_share"] = {k: float(share[np.where(Rh == v[3])[0]].sum()) for k, v in NM.items()}
    res["frac_time_coop"] = float(share[vis][sc[vis] >= 0.5].sum())
    top = vis[np.argsort(share[vis])[::-1][:15]]
    res["top"] = [dict(h=int(Rh[p]), share=float(share[p]), n=int(Rn[p]), coop=float(sc[p]), payoff=float(sp[p]),
                       name=NAME_OF.get(int(Rh[p]), ""), m=mstr(Rn[p], Ro[p], Rt[p])) for p in top]
    # ---- collapse analysis over ALL post-burn-in events (no cap)
    keep = lt >= burn
    seq = lpid[keep]; seqt = lt[keep]
    Vc = {}
    def V(pid):
        if pid not in Vc:
            Vc[pid] = vuln(*padm(Rn[pid], Ro[pid], Rt[pid]), prm)
        return Vc[pid]
    n_up = n_down = 0
    founder = None
    coll = []
    for k in range(1, len(seq)):
        a_, b_ = seq[k - 1], seq[k]
        if sc[a_] < 0.5 <= sc[b_]:
            founder = b_; n_up += 1
        if sc[a_] >= 0.5 > sc[b_]:
            n_down += 1
            coll.append((founder, int(a_)))
            founder = None
    res["n_up"] = n_up; res["n_down"] = n_down
    known = [(f, p) for f, p in coll if f is not None]
    if known:
        vpre = np.array([V(p) for _, p in known]); vfo = np.array([V(f) for f, _ in known])
        rob = vfo < 1; vul = vpre >= 1
        res["collapse"] = dict(n=len(coll), n_known_founder=len(known),
                               frac_pre_vulnerable=float(vul.mean()), frac_founder_robust=float(rob.mean()),
                               frac_drift=float((rob & vul).mean()),
                               frac_drift_given_robust=float((rob & vul).sum() / max(rob.sum(), 1)),
                               frac_pre_is_founder=float(np.mean([f == p for f, p in known])))
    else:
        res["collapse"] = dict(n=len(coll), n_known_founder=0)
    ep = episodes(seqt, seq, sc, burn, nsteps)
    for k_, v_ in ep.items():
        v_ = np.array(v_, float)
        res["ep_" + k_] = dict(n=int(len(v_)), mean=float(v_.mean()) if len(v_) else None,
                               median=float(np.median(v_)) if len(v_) else None)
    # ALLD sojourns from the log (time between entering and leaving ALLD); censored ends excluded
    soj = []
    if len(seq):
        isD = Rh[seq] == HD
        st = None
        for k in range(1, len(seq)):
            if isD[k] and not isD[k - 1]:
                st = seqt[k]
            elif (not isD[k]) and isD[k - 1] and st is not None:
                soj.append(seqt[k] - st)
                st = None
    res["alld_sojourn"] = dict(n=len(soj), mean=float(np.mean(soj)) if soj else None,
                               median=float(np.median(soj)) if soj else None)
    # ---- neighbourhoods of recorded (occupancy-weighted) residents
    nD, oD, tD = padm(*NM["ALLD"][:3])
    rng = np.random.default_rng(seed + 7)
    def nb_of(idx):
        X = []
        for i in idx:
            X.append(C.nbhd2(enc, size, a, recg[i].astype(np.int64), feat, prm["eps"], prm["delta"], R, S, T, P,
                             prm["N"], prm["beta"], prm["rule"], nD, oD, tD))
        return np.array(X)
    recs = [i for i in range(len(recp)) if i * rec >= burn]
    coopi = [i for i in recs if sc[recp[i]] >= 0.8]
    alldi = [i for i in recs if Rh[recp[i]] == HD]
    nmax = task.get("nbhd_max", 60)
    for lab, idx in [("nbhd_coop", coopi), ("nbhd_alld", alldi)]:
        if len(idx) > nmax:
            idx = list(rng.choice(idx, nmax, replace=False))
        if idx and task.get("nbhd", True):
            X = nb_of(idx)
            res[lab] = dict(n=len(idx), mean=X.mean(0).tolist(), sd=X.std(0).tolist())
    res["seconds_total"] = time.time() - t0
    return res


def do_sample2(task, outdir):
    enc, size, M, seed = task["enc"], task["size"], task["M"], task["seed"]
    feat = FEATS[task.get("feat", "M1" if enc >= 8 else "D")]
    prm = prm_of(task); R, S, T, P = prm["pay"]
    a = C.gene_alphabet2(enc, size)
    t0 = time.time()
    cnt, Rn, Ro, Rt, Rh, over, coll = C.sample_map2(enc, size, a, M, seed, feat, M + 10)
    coop, pay = C.selfplay_all(Rn, Ro, Rt, prm["eps"], prm["delta"], R, S, T, P)
    p = cnt / M
    def P_of(h):
        k = np.where(Rh == h)[0]
        return int(cnt[k].sum())
    res = dict(task=task, M=M, n_distinct=int(len(cnt)), n_singletons=int((cnt == 1).sum()), collisions=int(coll),
               overflow=int(over), prior_coop=float((p * coop).sum()),
               mass_by_n={int(k): float(p[Rn == k].sum()) for k in np.unique(Rn)},
               named_counts={k: P_of(v[3]) for k, v in NM.items()},
               good_turing_unseen=float((cnt == 1).sum() / M), seconds=time.time() - t0)
    if task.get("save"):
        klz = C.complexities(Rn, Ro, Rt, 4)
        np.savez_compressed(os.path.join(outdir, "phen_%s.npz" % task["id"]), keys=Rh, cnt=cnt, n=Rn, coop=coop, M=M, klz=klz)
    return res


def do_tables(task, outdir):
    enc, size, M, seed = task["enc"], task["size"], task["M"], task["seed"]
    feat = FEATS[task.get("feat", "D")]
    a = C.gene_alphabet2(enc, size)
    t0 = time.time()
    cnt = C.sample_tables(enc, size, a, M, seed, feat, task.get("nbits", 21))
    np.savez_compressed(os.path.join(outdir, "tables_%s.npz" % task["id"]), cnt=cnt.astype(np.int64), M=M)
    return dict(task=task, M=M, support_seen=int((cnt > 0).sum()), seconds=time.time() - t0)


def do_progenum(task, outdir):
    t0 = time.time()
    L = task["size"]
    cnt = C.enumerate_prog_tables(L, task["first"], C.FEAT, 6, C.FULL21, 0, 12)
    np.savez_compressed(os.path.join(outdir, "progenum_%s.npz" % task["id"]), cnt=cnt)
    return dict(task=task, support=int((cnt > 0).sum()), seconds=time.time() - t0)


def do_focal2(task, outdir):
    prm = prm_of(task); R, S, T, P = prm["pay"]
    enc, size = task["enc"], task["size"]
    feat = FEATS[task.get("feat", "D")]
    a = C.gene_alphabet2(enc, size)
    nD, oD, tD = padm(*NM["ALLD"][:3])
    res = dict(task=task)
    for nm_ in task["focal"]:
        G, tries = C.sample_network2(enc, size, a, NM[nm_][3], task["K"], task["maxtries"], task["seed"], feat)
        if len(G) == 0:
            res[nm_] = dict(n=0, tries=int(tries)); continue
        X = np.array([C.nbhd2(enc, size, a, g, feat, prm["eps"], prm["delta"], R, S, T, P, prm["N"], prm["beta"], prm["rule"], nD, oD, tD) for g in G])
        res[nm_] = dict(n=int(len(G)), tries=int(tries), mean=X.mean(0).tolist(), sd=X.std(0).tolist())
    return res


def do_ibm(task, outdir):
    prm = prm_of(task); R, S, T, P = prm["pay"]
    enc, size, seed = task["enc"], task["size"], task["seed"]
    feat = FEATS[task.get("feat", "D")]
    a = C.gene_alphabet2(enc, size)
    rng = np.random.default_rng(seed)
    g0 = np.array([rng.integers(x) for x in a], np.int64)
    t0 = time.time()
    r = C.ibm_run(enc, size, a, g0, prm["N"], task["steps"], task["burn"], task["mu"], prm["beta"], R, S, T, P,
                  prm["eps"], prm["delta"], seed, feat, task.get("cap", 3000000), task.get("sample_every", 50))
    return dict(task=task, coop=r[0], mono=r[1], mean_K=r[2], n_mut=int(r[3]), overflow=int(r[4]), collisions=int(r[5]),
                n_pheno=int(r[6]), seconds=time.time() - t0)


KINDS = dict(chain2=do_chain2, sample2=do_sample2, tables=do_tables, progenum=do_progenum, focal2=do_focal2, ibm=do_ibm)


def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        tasks = json.load(stream)
    outdir, wid = sys.argv[2:4]
    missing = sorted({path for t in tasks for path in t.get("requires", []) if not os.path.isfile(path)})
    if missing:
        raise SystemExit("Missing task inputs: " + ", ".join(missing))
    os.makedirs(os.path.join(outdir, "claims"), exist_ok=True)
    tasks.sort(key=lambda t: -t.get("cost", 1))
    failed = False
    for t in tasks:
        fout = os.path.join(outdir, "%s_%s.json" % (t["kind"], t["id"]))
        if os.path.exists(fout):
            continue
        lock = os.path.join(outdir, "claims", t["id"] + ".lock")
        try:
            claim = open(lock, "x", encoding="utf-8")
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            with claim:
                claim.write(wid)
            if os.path.exists(fout):
                continue
            res = KINDS[t["kind"]](t, outdir)
            with open(fout + ".tmp", "w", encoding="utf-8") as stream:
                json.dump(res, stream)
            os.replace(fout + ".tmp", fout)
            print(wid, "done", t["id"], round(time.time() - t0), flush=True)
        except Exception:
            failed = True
            with open(os.path.join(outdir, "ERR_%s.txt" % t["id"]), "w", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
            print(wid, "ERR", t["id"], flush=True)
        finally:
            os.remove(lock)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
