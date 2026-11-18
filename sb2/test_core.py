"""test_core.py -- independent checks of the core routines (run: python test_core.py).
1. Discounted payoffs and cooperation from play() against a direct Monte Carlo simulation of the repeated game.
2. Canonical phenotypes are invariant under random relabelling of automaton states (minimisation + BFS relabelling).
3. Fixation probabilities: neutral case equals 1/N; constant-fitness-difference case matches the closed form.
4. Phenotype of every memory-one table equals the machine built from its code (compiler vs reference construction).
"""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb_core2 as C

rng = np.random.default_rng(7)
PAY = (2.0, -1.0, 3.0, 0.0); EPS = 0.01; DELTA = 0.9


def rand_machine(k):
    g = np.concatenate([rng.integers(0, 2, k), rng.integers(0, k, 4 * k)]).astype(np.int64)
    return C.phen_of2(4, k, g, C.FEAT), g


def pad(n, o, t):
    oo = np.zeros(C.NMAX, np.int64); tt = np.zeros((C.NMAX, 4), np.int64)
    oo[:n] = o[:n]; tt[:n] = t[:n]
    return n, oo, tt


def mc_play(A, B, T=200000):
    """Monte Carlo: random game length (geometric with continuation delta) gives the discounted average directly."""
    nA, oA, tA = A; nB, oB, tB = B
    R, S, Tt, P = PAY; U = {(1, 1): (R, R), (1, 0): (S, Tt), (0, 1): (Tt, S), (0, 0): (P, P)}
    ua = ca = 0.0; nr = 0
    for _ in range(T // 10):
        sa = sb = 0
        while True:
            a = oA[sa] ^ (rng.random() < EPS); b = oB[sb] ^ (rng.random() < EPS)
            ua += U[(int(a), int(b))][0]; ca += a; nr += 1
            sa = tA[sa, 2 * a + b]; sb = tB[sb, 2 * b + a]
            if rng.random() > DELTA:
                break
    return ua / nr, ca / nr


ok = True
# 1. payoffs
for trial in range(4):
    (n1, o1, t1, h1), _ = rand_machine(3)
    (n2, o2, t2, h2), _ = rand_machine(3)
    A = pad(n1, o1, t1); B = pad(n2, o2, t2)
    uA, uB, cA, cB = C.play(*A, *B, EPS, DELTA, *PAY, 1e-12)
    mu, mc = mc_play(A, B)
    good = abs(mu - uA) < 0.03 and abs(mc - cA) < 0.01
    ok &= good
    print("payoff check", trial, round(uA, 4), round(mu, 4), round(cA, 4), round(mc, 4), "ok" if good else "FAIL")
# 2. relabelling invariance
for trial in range(200):
    k = 4
    (n, o, t, h), g = rand_machine(k)
    perm = np.concatenate([[0], 1 + rng.permutation(k - 1)])
    g2 = g.copy()
    for s in range(k):
        g2[perm[s]] = g[s]
        for oc in range(4):
            g2[k + 4 * perm[s] + oc] = perm[g[k + 4 * s + oc]]
    n2, o2, t2, h2 = C.phen_of2(4, k, g2, C.FEAT)
    if h2 != h:
        ok = False; print("relabel FAIL", trial)
print("relabelling invariance: 200 random 4-state automata", "ok" if ok else "FAIL")
# 3. fixation probabilities
N = 50
r0 = C.fixprob(1.0, 1.0, 1.0, 1.0, N, 1.0, 0)
good = abs(r0 - 1 / N) < 1e-12
d = 0.05; beta = 1.0
r1 = C.fixprob(1.0, 1.0, 1.0 + d, 1.0 + d, N, beta, 0)
q = np.exp(-beta * d); closed = (1 - q) / (1 - q ** N)
good &= abs(r1 - closed) < 1e-10
ok &= good
print("fixation checks", r0, r1, closed, "ok" if good else "FAIL")
# 4. memory-one compiler
bad = 0
for code in range(32):
    g = np.array([(code >> i) & 1 for i in range(5)], np.int64)
    n, o, t, h = C.phen_of2(0, 0, g, C.FEAT)
    n2, o2, t2 = C.memory_one(g[0], g[4], g[3], g[2], g[1])
    co = np.zeros(C.NMAX, np.int64); ct = np.zeros((C.NMAX, 4), np.int64)
    nc = C.minimise(n2, o2, t2, co, ct)
    if C.mhash(nc, co, ct) != h:
        bad += 1
print("memory-one compiler vs reference:", 32 - bad, "of 32 agree")
ok &= bad == 0
print("ALL OK" if ok else "SOME CHECKS FAILED")

if not ok:
    raise SystemExit(1)
