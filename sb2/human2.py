"""human2.py -- revised analysis of the memory-one strategy elicitation of Dal Bo & Frechette (2019, AER),
online appendix Table A5 (246 participants, 5 treatments).

Complexity measures
  m_full : states of the minimal Moore machine on all histories (our phenotype definition; ordering invariant)
  m_path : states of the smallest machine that agrees with the strategy on every history reachable when the
           player itself makes no implementation error (DBF's AC'/AD' convention; ordering invariant)
  klz_o  : Lempel-Ziv complexity of the 341-bit fingerprint for each of the 24 outcome orderings (and their mean)
Choice models (conditional logit over the 32 memory-one strategies, separately normalised per treatment)
  U_t(s) = -gamma * m(s) + lambda * u_t(s) [+ focal dummies]
  u_t(s) = expected discounted payoff of s against the treatment's empirical strategy distribution (no errors),
           divided by T-S so that lambda is comparable across treatments.
Usage: python human2.py <dbf_csv> <outjson>
"""
import os, sys, json, itertools
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_core2 as C

TREAT = {"d050_R32": (0.5, 32.0), "d050_R48": (0.5, 48.0), "d075_R32": (0.75, 32.0), "d075_R48": (0.75, 48.0), "d090_R32": (0.9, 32.0)}
S_, T_, P_ = 12.0, 50.0, 25.0
CODES = ["".join(x) for x in itertools.product("CD", repeat=5)]   # p0, after CC, DC, CD, DD (own action first)


def bits(code):
    return [1 if ch == "C" else 0 for ch in code]


def machine(code):
    p0, pCC, pDC, pCD, pDD = bits(code)
    n, o, t = C.memory_one(p0, pCC, pCD, pDC, pDD)
    co = np.zeros(C.NMAX, np.int64); ct = np.zeros((C.NMAX, 4), np.int64)
    nc = C.minimise(n, o, t, co, ct)
    return nc, co, ct


def onpath_states(code):
    """minimum state count over all completions of the responses at own-unreachable contexts."""
    b = bits(code); resp = {(1, 1): b[1], (0, 1): b[2], (1, 0): b[3], (0, 0): b[4]}   # (own, opp) -> action
    reach = {b[0]}; changed = True
    while changed:
        changed = False
        for a in list(reach):
            for o in (0, 1):
                if resp[(a, o)] not in reach:
                    reach.add(resp[(a, o)]); changed = True
    free = [k for k in resp if k[0] not in reach]
    best = 99
    for fill in itertools.product((0, 1), repeat=len(free)):
        r = dict(resp); r.update(dict(zip(free, fill)))
        c = "".join("C" if x else "D" for x in [b[0], r[(1, 1)], r[(0, 1)], r[(1, 0)], r[(0, 0)]])
        best = min(best, machine(c)[0])
    return best


def klz_order(nc, co, ct, perm):
    """relabel the 4 outcomes by perm (outcome o -> perm[o]) before fingerprinting."""
    inv = np.argsort(perm)
    ct2 = ct.copy()
    for s in range(nc):
        for o in range(4):
            ct2[s, perm[o]] = ct[s, o]
    return C.k_lz(C.fingerprint(nc, co, ct2, 4))


def payoffs(mach, delta, R):
    K = len(mach); U = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            uA, uB, _, _ = C.play(*mach[i], *mach[j], 0.0, delta, R, S_, T_, P_, 1e-12)
            U[i, j] = uA; U[j, i] = uB
    return U / (T_ - S_)


def fit(cnt, X):
    """conditional logit; cnt: (T, 32) counts; X: (T, 32, p) covariates. returns coef, loglik."""
    p = X.shape[2]
    def nll(b):
        v = X @ b; v = v - v.max(axis=1, keepdims=True)
        lp = v - np.log(np.exp(v).sum(axis=1, keepdims=True))
        return -(cnt * lp).sum()
    if p == 0:
        return np.zeros(0), float((cnt * np.log(1 / X.shape[1])).sum())
    o = minimize(nll, np.zeros(p), method="BFGS")
    return o.x, float(-o.fun)


def main():
    dbf = pd.read_csv(sys.argv[1])
    ncols = ["n_" + t for t in TREAT]
    cnt = dbf.set_index("code")[ncols].reindex(CODES).fillna(0).values.T.astype(float)   # (5, 32)
    assert cnt.sum() == 246, cnt.sum()
    mach = [machine(c) for c in CODES]
    m_full = np.array([m[0] for m in mach], float)
    m_path = np.array([onpath_states(c) for c in CODES], float)
    perms = list(itertools.permutations(range(4)))
    KLZ = np.array([[klz_order(*m, np.array(pm)) for m in mach] for pm in perms])   # (24, 32)
    klz_mean = KLZ.mean(axis=0)
    ident = perms.index((0, 1, 2, 3))
    # expected payoffs vs the empirical population of each treatment
    u = np.zeros((5, 32)); Umats = {}
    for k, (t, (d, R)) in enumerate(TREAT.items()):
        U = payoffs(mach, d, R); Umats[t] = U
        pop = cnt[k] / cnt[k].sum()
        u[k] = U @ pop
    out = dict(codes=CODES, m_full=m_full.tolist(), m_path=m_path.tolist(), klz_mean=klz_mean.tolist(),
               klz_identity=KLZ[ident].tolist(), counts=cnt.tolist(), u=u.tolist())
    h = cnt.sum(axis=0)
    out["share_one_state_full"] = float(h[m_full == 1].sum() / h.sum())
    out["share_one_state_path"] = float(h[m_path == 1].sum() / h.sum())
    out["prior_one_state_uniform_full"] = float((m_full == 1).mean())
    out["prior_one_state_uniform_path"] = float((m_path == 1).mean())
    T = cnt.shape[0]
    def X_of(cols):
        return np.stack(cols, axis=2) if cols else np.zeros((T, 32, 0))
    rep = lambda v: np.repeat(v[None, :], T, axis=0)
    models = {}
    specs = {
        "uniform": [],
        "m_full": [-rep(m_full)], "m_path": [-rep(m_path)], "klz_mean": [-rep(klz_mean / 10)],
        "payoff": [u],
        "payoff+m_full": [u, -rep(m_full)], "payoff+m_path": [u, -rep(m_path)], "payoff+klz_mean": [u, -rep(klz_mean / 10)],
    }
    for nm, cols in specs.items():
        b, ll = fit(cnt, X_of(cols)); models[nm] = dict(coef=b.tolist(), ll=ll, k=len(cols))
    for a_, b_ in [("uniform", "m_full"), ("uniform", "m_path"), ("uniform", "klz_mean"), ("uniform", "payoff"),
                   ("payoff", "payoff+m_full"), ("payoff", "payoff+m_path"), ("payoff", "payoff+klz_mean")]:
        d = models[b_]["ll"] - models[a_]["ll"]
        models[b_]["vs_" + a_] = dict(gain=d, p=float(chi2.sf(2 * d, models[b_]["k"] - models[a_]["k"])))
    out["models"] = models
    # treatment-specific gamma for m_full with and without payoff control
    out["by_treatment"] = {}
    for k, t in enumerate(TREAT):
        c1 = cnt[k:k + 1]
        b0, l0 = fit(c1, np.zeros((1, 32, 0)))
        b1, l1 = fit(c1, (-m_full)[None, :, None])
        b2, l2 = fit(c1, u[k:k + 1, :, None])
        b3, l3 = fit(c1, np.stack([u[k:k + 1], -m_full[None, :]], axis=2))
        out["by_treatment"][t] = dict(n=int(c1.sum()), gamma=float(b1[0]), gamma_ctrl=float(b3[1]), lam_ctrl=float(b3[0]),
                                      p_gamma_ctrl=float(chi2.sf(2 * (l3 - l2), 1)))
    # LZ ordering sensitivity among the two-state strategies (identity ordering = v1 analysis)
    two = m_full == 2
    gam = []
    for q in range(24):
        b, ll = fit(h[two][None, :], (-KLZ[q][two] / 10)[None, :, None] * np.ones((1, 1, 1)))
        b0, l0 = fit(h[two][None, :], np.zeros((1, int(two.sum()), 0)))
        gam.append(dict(order=list(perms[q]), gamma_per_LZ=float(b[0] / 10), gain=float(ll - l0)))
    out["lz_orderings_two_state"] = gam
    # leave-focal-out on the two-state subset with identity ordering
    lfo = {}
    for drop in [[], ["CCCDD"], ["CCCDD", "DCCDD"]]:
        keep = two & ~np.isin(CODES, drop)
        b, ll = fit(h[keep][None, :], (-KLZ[ident][keep] / 10)[None, :, None])
        _, l0 = fit(h[keep][None, :], np.zeros((1, int(keep.sum()), 0)))
        lfo["drop_" + ("+".join(drop) or "none")] = dict(gamma_per_LZ=float(b[0] / 10), gain=float(ll - l0), p=float(chi2.sf(2 * (ll - l0), 1)))
    out["leave_focal_out"] = lfo
    # participant bootstrap within treatments (no session identifiers in Table A5)
    rng = np.random.default_rng(2019); B = 2000; boot = {k: [] for k in ["m_full", "payoff+m_full_gamma", "payoff+m_path_gamma"]}
    for _ in range(B):
        cb = np.zeros_like(cnt)
        for k in range(T):
            nk = int(cnt[k].sum()); cb[k] = rng.multinomial(nk, cnt[k] / nk)
        ub = np.zeros_like(u)
        for k, t in enumerate(TREAT):
            ub[k] = Umats[t] @ (cb[k] / cb[k].sum())
        boot["m_full"].append(fit(cb, X_of([-rep(m_full)]))[0][0])
        boot["payoff+m_full_gamma"].append(fit(cb, X_of([ub, -rep(m_full)]))[0][1])
        boot["payoff+m_path_gamma"].append(fit(cb, X_of([ub, -rep(m_path)]))[0][1])
    out["bootstrap"] = {k: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), frac_pos=float(np.mean(np.array(v) > 0)))
                        for k, v in boot.items()}
    # rank association within the two-state subset (identity and mean-over-orderings LZ)
    out["spearman_two_state"] = dict(identity=[float(x) for x in spearmanr(KLZ[ident][two], h[two])],
                                     mean_order=[float(x) for x in spearmanr(klz_mean[two], h[two])])
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
