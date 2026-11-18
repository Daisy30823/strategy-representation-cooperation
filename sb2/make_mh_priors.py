"""make_mh_priors.py -- convert truth-table counts (tables_*.npz, indexed by the 21-bit memory-two table)
into phenotype-keyed prior files for Metropolis-Hastings reweighting (keys = canonical machine hash).
Usage: python make_mh_priors.py <rundir> <outdir>
Writes  prior_<VAR>_<label>.npz  with keys, cnt, M for:  s11, s12, s13 (1e7 each), big (1e9 = 10 x 1e8),
and a summary json with support/Good-Turing statistics.
"""
import os, sys, json, glob
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_core2 as C

rundir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
ns, hs = C.enumerate_table2(C.FEAT)
assert len(np.unique(hs)) == len(hs), "memory-two table map is not injective"
summ = {}
for var in ["CIRC16", "PROG16"]:
    sets = {}
    for s in [11, 12, 13]:
        f = os.path.join(rundir, "tables_tab__%s__s%d.npz" % (var, s))
        if os.path.exists(f):
            sets["s%d" % s] = np.load(f)["cnt"].astype(np.int64)
    big = [np.load(f)["cnt"].astype(np.int64) for f in sorted(glob.glob(os.path.join(rundir, "tables_tab__%s__big*.npz" % var)))]
    if big:
        sets["big"] = np.sum(big, axis=0)
        sets["big1"] = big[0]
    for lab, cnt in sets.items():
        M = int(cnt.sum())
        np.savez_compressed(os.path.join(outdir, "prior_%s_%s.npz" % (var, lab)), keys=hs, cnt=cnt, M=M)
        summ["%s_%s" % (var, lab)] = dict(M=M, seen=int((cnt > 0).sum()), singletons=int((cnt == 1).sum()),
                                          good_turing_unseen_mass=float((cnt == 1).sum() / M),
                                          mass_le2=float(cnt[ns <= 2].sum() / M), n_le2_seen=int(((cnt > 0) & (ns <= 2)).sum()))
json.dump(summ, open(os.path.join(outdir, "priors_summary.json"), "w"), indent=1)
print(summ)
