"""exact1.py -- exact small-mutation-limit benchmark on complete genotype spaces.

For each encoding the whole genotype space is enumerated, every genotype is compiled to its minimal
Moore machine, and the embedded mutation-proposal chain of the small-mutation limit is built as a sparse
matrix and solved for its stationary distribution (no simulation).  Four dynamics are computed:
  point   : uniform point mutation (one gene, alternative allele uniform), acceptance rho(x,x')
  point_mh: as point, acceptance rho(x,x') * min{1, P_E(x)/P_E(x')} with the EXACT encoding prior P_E
  hoc     : house-of-cards mutation (whole genotype redrawn uniformly), i.e. proposal x' ~ P_E
  hoc_mh  : house-of-cards with the same exact Metropolis factor (proposal becomes uniform on the support)
Usage: python exact1.py <outdir> <spec>   spec = NAME:enc:size  (e.g. CIRC1:8:3)
"""
import os, sys, json, time, itertools
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_core2 as C

PAY = (2.0, -1.0, 3.0, 0.0); EPS = 0.01; DELTA = 0.9; N = 50
BETAS = [0.0, 0.1, 0.3, 1.0, 3.0]


def enumerate_space(enc, size, feat):
    a = C.gene_alphabet2(enc, size).astype(np.int64)
    n = int(np.prod(a))
    stride = np.ones(len(a), np.int64)
    for l in range(len(a) - 2, -1, -1):
        stride[l] = stride[l + 1] * a[l + 1]
    reg = {}; mach = []; ph = np.zeros(n, np.int64)
    for i, g in enumerate(itertools.product(*[range(x) for x in a])):
        nc, o, t, h = C.phen_of2(enc, size, np.array(g, np.int64), feat)
        k = reg.get(h)
        if k is None:
            k = len(mach); reg[h] = k; mach.append((nc, o.copy(), t.copy()))
        ph[i] = k
    return a, stride, ph, mach


def payoff_matrix(mach):
    K = len(mach)
    def pad(m):
        o = np.zeros(C.NMAX, np.int64); t = np.zeros((C.NMAX, 4), np.int64)
        o[:m[0]] = m[1]; t[:m[0]] = m[2]; return m[0], o, t
    MM = [pad(m) for m in mach]
    U = np.zeros((K, K)); coop = np.zeros(K)
    for i in range(K):
        for j in range(i, K):
            uA, uB, cA, cB = C.play(*MM[i], *MM[j], EPS, DELTA, *PAY, 1e-12)
            U[i, j] = uA; U[j, i] = uB
            if i == j:
                coop[i] = 0.5 * (cA + cB)
    return U, coop


def rho_matrix(U, beta):
    K = len(U); R = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            R[i, j] = 1.0 / N if i == j else C.fixprob(U[i, i], U[i, j], U[j, i], U[j, j], N, beta, 0)
    return R


def stationary(P, x0=None):
    """left stationary vector of a sparse row-stochastic matrix (irreducible on its support):
    fix pi_0 = 1 and solve the reduced system (I - P^T)[1:,1:] x = P^T[1:,0] (dense LAPACK for
    n <= 3000, otherwise restarted GMRES); the residual |pi P - pi| is checked."""
    n = P.shape[0]
    PT = sp.csr_matrix(P).T.tocsr()
    if n <= 7000:
        A = (np.eye(n) - PT.toarray()); A[-1, :] = 1.0
        b = np.zeros(n); b[-1] = 1.0
        pi = np.linalg.solve(A, b)
    else:
        A = (sp.identity(n, format="csr") - PT)[1:, 1:].tocsr()
        b = np.asarray(PT[1:, 0].todense()).ravel()
        x, info = spla.gmres(A, b, x0=None if x0 is None else x0[1:] / x0[0], rtol=1e-12, atol=0, restart=300, maxiter=20000)
        if info != 0:
            raise RuntimeError("gmres info=%d" % info)
        pi = np.concatenate([[1.0], x])
    res = np.abs(pi @ P - pi).max() / np.abs(pi).max()
    assert res < 1e-9, res
    return pi


def point_matrix(a, stride, ph, R, w_mh=None):
    n = len(ph); L = len(a)
    idx = np.arange(n, dtype=np.int64)
    rows = []; cols = []; vals = []
    for l in range(L):
        gl = (idx // stride[l]) % a[l]
        for v in range(a[l]):
            m = gl != v
            i = idx[m]; j = i + (v - gl[m]) * stride[l]
            p = R[ph[i], ph[j]] / (L * (a[l] - 1))
            if w_mh is not None:
                p = p * np.minimum(1.0, w_mh[ph[i]] / w_mh[ph[j]])
            rows.append(i); cols.append(j); vals.append(p)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    d = 1.0 - np.asarray(P.sum(axis=1)).ravel()
    return (P + sp.diags(d)).tocsr()


def hoc_matrix(prior, R, mh=False):
    K = len(prior)
    P = R * prior[None, :]
    if mh:
        P = P * np.minimum(1.0, prior[:, None] / prior[None, :])
    np.fill_diagonal(P, 0.0)
    P[np.arange(K), np.arange(K)] = 1.0 - P.sum(axis=1)
    return P


def main():
    outdir, spec = sys.argv[1], sys.argv[2]
    name, enc, size = spec.split(":"); enc = int(enc); size = int(size)
    feat = C.FEAT_M1 if enc >= 8 else C.FEAT
    t0 = time.time()
    a, stride, ph, mach = enumerate_space(enc, size, feat)
    n = len(ph); K = len(mach)
    U, coop = payoff_matrix(mach)
    prior = np.bincount(ph, minlength=K) / n
    names = C.named_machines(); hname = {}
    for nm, v in names.items():
        hname[v[3]] = nm
    keys = [C.mhash(m[0], np.pad(m[1], (0, C.NMAX - m[0])), np.pad(m[2], ((0, C.NMAX - m[0]), (0, 0)))) for m in mach]
    lab = [hname.get(k, "") for k in keys]
    res = dict(name=name, enc=enc, size=size, n_geno=n, n_pheno=K, alphabet=a.tolist(),
               prior_coop=float(prior @ coop), uniform_coop=float(coop.mean()),
               phen=[dict(label=lab[k], n=int(mach[k][0]), out=mach[k][1].tolist(), tr=mach[k][2].tolist(),
                          coop=float(coop[k]), prior=float(prior[k])) for k in range(K)], dyn={})
    warm = {}
    for beta in BETAS:
        R = rho_matrix(U, beta)
        for mode in ["point", "point_mh", "hoc", "hoc_mh"]:
            if mode.startswith("point") and beta == 0.0:
                # exact closed forms at beta=0 (Proposition 2 and the reweighting proposition):
                # uniform over genotypes, or uniform over the reachable phenotypes
                pi = prior.copy() if mode == "point" else np.ones(K) / K
            elif mode.startswith("point"):
                P = point_matrix(a, stride, ph, R, prior if mode == "point_mh" else None)
                pi_g = stationary(P, warm.get(mode))
                warm[mode] = pi_g
                pi = np.bincount(ph, weights=pi_g, minlength=K)
            else:
                P = hoc_matrix(prior, R, mode == "hoc_mh")
                pi = stationary(sp.csr_matrix(P))
            pi = np.maximum(pi, 0); pi /= pi.sum()
            res["dyn"]["%s_b%g" % (mode, beta)] = dict(coop=float(pi @ coop), pi=pi.tolist())
        print(name, beta, {m: round(res["dyn"]["%s_b%g" % (m, beta)]["coop"], 4) for m in ["point", "point_mh", "hoc", "hoc_mh"]}, flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(outdir, "exact1_%s.json" % name), "w"))


if __name__ == "__main__":
    main()
