"""
sb_core2.py -- extensions used in the revision (v2).
Imports the v1 core unchanged and adds:
  * further encodings: valence-mirrored controls and small exactly enumerable memory-one encodings
      6  PROG-defC  stack program whose empty stack yields cooperation (otherwise as enc 3)
      7  RNN-tieC   recurrent threshold network whose units fire on z >= 0 (otherwise as enc 5)
      8  CIRC1      Boolean circuit over the memory-one features (a1, b1, first-round flag)
      9  PROG1      stack program over the memory-one features (9 instructions)
     10  PERC1      single threshold unit over the memory-one features (4 weights in -2..2)
     11  RNN1t      one-unit recurrent threshold network with ternary weights (-1,0,1)
    Feature padding (value used for missing history) is chosen through the feature array:
    FEAT (pad D, as in v1) or FEAT_C (pad C).
  * run_chain2: the small-mutation-limit chain with point mutation (mode 0, identical random-number
    stream to v1 run_chain) or house-of-cards mutation (mode 1: a fresh uniformly random genotype),
    window means, occupancy bookkeeping for ALLD and ALLC, and a canonical-form collision check.
  * table-count samplers for encodings that compile into the memory-two universe.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict
from sb_core import *          # v1 core (unchanged)
from sb_core import _register, _load
import sb_core as _c1

FULL5 = 31


def _feature_masks_pad(pad):
    """memory-two feature masks; missing-history actions are coded as `pad` (0 = D, 1 = C)."""
    F = np.zeros(6, np.int64)
    for k in range(21):
        if k == 0:
            f = (pad, pad, pad, pad, 1, 0)
        elif k <= 4:
            o1 = k - 1
            f = (o1 >> 1, o1 & 1, pad, pad, 0, 1)
        else:
            idx = k - 5
            o2, o1 = idx >> 2, idx & 3
            f = (o1 >> 1, o1 & 1, o2 >> 1, o2 & 1, 0, 0)
        for i in range(6):
            if f[i]:
                F[i] |= (1 << k)
    return F


FEAT_C = _feature_masks_pad(1)
assert np.array_equal(_feature_masks_pad(0), FEAT)


def _feature_masks_m1():
    """memory-one features (a1, b1, first-round flag) over the 5 memory-one contingencies:
    bit 0 = first round, bit 1+o = after outcome o = 2*own + opp."""
    F = np.zeros(3, np.int64)
    for k in range(5):
        if k == 0:
            f = (0, 0, 1)
        else:
            o = k - 1
            f = (o >> 1, o & 1, 0)
        for i in range(3):
            if f[i]:
                F[i] |= (1 << k)
    return F


FEAT_M1 = _feature_masks_m1()


def gene_alphabet2(enc, size):
    if enc <= 5:
        return gene_alphabet(enc, size)
    if enc == 6:
        return np.full(size, 12, np.int64)
    if enc == 7:
        return gene_alphabet(5, size)
    if enc == 8:
        a = []
        for j in range(size):
            a += [4, 3 + j, 3 + j]
        return np.array(a, np.int64)
    if enc == 9:
        return np.full(size, 9, np.int64)
    if enc == 10:
        return np.full(4, 5, np.int64)
    if enc == 11:
        return np.array([3] * 6 + [2], np.int64)
    raise ValueError(enc)


@njit(cache=True)
def circ_gen(g, G, feat, nf, full):
    vals = np.zeros(nf + G, np.int64)
    for i in range(nf):
        vals[i] = feat[i]
    for j in range(G):
        t = g[3 * j]
        x = vals[g[3 * j + 1]]
        y = vals[g[3 * j + 2]]
        if t == 0:
            v = x & y
        elif t == 1:
            v = x | y
        elif t == 2:
            v = (~(x & y)) & full
        else:
            v = (~(x | y)) & full
        vals[nf + j] = v
    return vals[nf + G - 1]


@njit(cache=True)
def prog_gen(g, L, feat, nf, full, dflt):
    """stack program; instruction codes 0..nf-1 push a feature, then NOT, AND, OR, XOR, DUP, SWAP.
    Instructions with too few operands are skipped; an empty final stack returns `dflt`."""
    st = np.zeros(L + 1, np.int64)
    sp = 0
    for i in range(L):
        op = g[i]
        if op < nf:
            st[sp] = feat[op]
            sp += 1
        elif op == nf:
            if sp >= 1:
                st[sp - 1] = (~st[sp - 1]) & full
        elif op <= nf + 3:
            if sp >= 2:
                x = st[sp - 1]
                y = st[sp - 2]
                if op == nf + 1:
                    v = x & y
                elif op == nf + 2:
                    v = x | y
                else:
                    v = x ^ y
                st[sp - 2] = v
                sp -= 1
        elif op == nf + 4:
            if sp >= 1:
                st[sp] = st[sp - 1]
                sp += 1
        else:
            if sp >= 2:
                x = st[sp - 1]
                st[sp - 1] = st[sp - 2]
                st[sp - 2] = x
    if sp == 0:
        return dflt
    return st[sp - 1]


@njit(cache=True)
def table1_machine(tab, out, tr):
    out[0] = tab & 1
    for o in range(4):
        tr[0, o] = 1 + o
    for s in range(4):
        out[1 + s] = (tab >> (1 + s)) & 1
        for o in range(4):
            tr[1 + s, o] = 1 + o
    return 5


@njit(cache=True)
def perc_table(g):
    tab = 0
    for k in range(5):
        if k == 0:
            a, b, h = -1, -1, 1
        else:
            o = k - 1
            a = 2 * (o >> 1) - 1
            b = 2 * (o & 1) - 1
            h = -1
        z = (g[0] - 2) * a + (g[1] - 2) * b + (g[2] - 2) * h + (g[3] - 2)
        if z > 0:
            tab |= (1 << k)
    return tab


@njit(cache=True)
def rnn_compile(g, H, shift, tieC, out, tr):
    """recurrent threshold network; weights are gene - shift; unit fires if z > 0 (or z >= 0 if tieC)."""
    off_W = 0
    off_U = H * H
    off_c = off_U + 2 * H
    off_v = off_c + H
    off_d = off_v + H
    off_h = off_d + 1
    ns = 1 << H
    hb = np.zeros(H, np.int64)
    thr = -1 if tieC else 0
    for s in range(ns):
        for i in range(H):
            hb[i] = 2 * ((s >> i) & 1) - 1
        z = g[off_d] - shift
        for i in range(H):
            z += (g[off_v + i] - shift) * hb[i]
        out[s] = 1 if z > thr else 0
        for o in range(4):
            xa = 2 * (o >> 1) - 1
            xb = 2 * (o & 1) - 1
            nxt = 0
            for i in range(H):
                zi = g[off_c + i] - shift
                for j in range(H):
                    zi += (g[off_W + i * H + j] - shift) * hb[j]
                zi += (g[off_U + 2 * i] - shift) * xa + (g[off_U + 2 * i + 1] - shift) * xb
                if zi > thr:
                    nxt |= (1 << i)
            tr[s, o] = nxt
    s0 = 0
    for i in range(H):
        s0 |= (g[off_h + i] << i)
    if s0 != 0:
        for o in range(4):
            a = tr[0, o]
            tr[0, o] = tr[s0, o]
            tr[s0, o] = a
        a = out[0]
        out[0] = out[s0]
        out[s0] = a
        for s in range(ns):
            for o in range(4):
                if tr[s, o] == 0:
                    tr[s, o] = s0
                elif tr[s, o] == s0:
                    tr[s, o] = 0
    return ns


@njit(cache=True)
def compile_raw2(enc, size, g, feat, out, tr):
    if enc <= 1 or enc == 4:
        return compile_raw(enc, size, g, feat, out, tr)
    if enc == 2:
        return table2_machine(circ_gen(g, size, feat, 6, FULL21), out, tr)
    if enc == 3:
        return table2_machine(prog_gen(g, size, feat, 6, FULL21, 0), out, tr)
    if enc == 6:
        return table2_machine(prog_gen(g, size, feat, 6, FULL21, FULL21), out, tr)
    if enc == 5:
        return rnn_compile(g, size, 2, False, out, tr)
    if enc == 7:
        return rnn_compile(g, size, 2, True, out, tr)
    if enc == 8:
        return table1_machine(circ_gen(g, size, feat, 3, FULL5), out, tr)
    if enc == 9:
        return table1_machine(prog_gen(g, size, feat, 3, FULL5, 0), out, tr)
    if enc == 10:
        return table1_machine(perc_table(g), out, tr)
    return rnn_compile(g, 1, 1, False, out, tr)


@njit(cache=True)
def table_of(enc, size, g, feat):
    """21-bit (memory-two) or 5-bit (memory-one) truth table for the table-compiling encodings."""
    if enc == 1:
        tab = 0
        for k in range(21):
            if g[k]:
                tab |= (1 << k)
        return tab
    if enc == 2:
        return circ_gen(g, size, feat, 6, FULL21)
    if enc == 3:
        return prog_gen(g, size, feat, 6, FULL21, 0)
    if enc == 6:
        return prog_gen(g, size, feat, 6, FULL21, FULL21)
    if enc == 0:
        tab = 0
        for k in range(5):
            if g[k]:
                tab |= (1 << k)
        return tab
    if enc == 8:
        return circ_gen(g, size, feat, 3, FULL5)
    if enc == 9:
        return prog_gen(g, size, feat, 3, FULL5, 0)
    return perc_table(g)


@njit(cache=True)
def phen_of2(enc, size, g, feat):
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    n = compile_raw2(enc, size, g, feat, out, tr)
    nc = minimise(n, out, tr, cout, ctr)
    return nc, cout[:nc].copy(), ctr[:nc].copy(), mhash(nc, cout, ctr)


@njit(cache=True)
def _register_chk(n, cout, ctr, hsh, reg, R_n, R_o, R_t, R_h, nreg, coll):
    """as v1 _register, but verifies the canonical machine on a hash hit; coll[0] counts mismatches."""
    if hsh in reg:
        pid = reg[hsh]
        same = R_n[pid] == n
        if same:
            for s in range(n):
                if R_o[pid, s] != cout[s]:
                    same = False
                    break
                for o in range(4):
                    if R_t[pid, s, o] != ctr[s, o]:
                        same = False
                        break
                if not same:
                    break
        if not same:
            coll[0] += 1
        return pid, nreg
    pid = nreg
    if pid >= R_n.shape[0]:
        return -1, nreg
    reg[hsh] = pid
    R_n[pid] = n
    R_h[pid] = hsh
    for s in range(n):
        R_o[pid, s] = cout[s]
        for o in range(4):
            R_t[pid, s, o] = ctr[s, o]
    return pid, nreg + 1


@njit(cache=True)
def sample_map2(enc, size, asz, M, seed, feat, cap):
    np.random.seed(seed)
    L = asz.shape[0]
    g = np.zeros(L, np.int64)
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    reg = Dict.empty(key_type=types.int64, value_type=types.int64)
    R_n = np.zeros(cap, np.int64)
    R_o = np.zeros((cap, NMAX), np.int8)
    R_t = np.zeros((cap, NMAX, 4), np.int8)
    R_h = np.zeros(cap, np.int64)
    cnt = np.zeros(cap, np.int64)
    coll = np.zeros(1, np.int64)
    nreg = 0
    over = 0
    for m in range(M):
        for i in range(L):
            g[i] = np.random.randint(asz[i])
        n = compile_raw2(enc, size, g, feat, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        hs = mhash(nc, cout, ctr)
        pid, nreg2 = _register_chk(nc, cout, ctr, hs, reg, R_n, R_o, R_t, R_h, nreg, coll)
        if pid < 0:
            over += 1
            continue
        nreg = nreg2
        cnt[pid] += 1
    return (cnt[:nreg].copy(), R_n[:nreg].copy(), R_o[:nreg].copy(), R_t[:nreg].copy(), R_h[:nreg].copy(),
            over, coll[0])


@njit(cache=True)
def sample_tables(enc, size, asz, M, seed, feat, nbits):
    """counts of truth tables (2**nbits) produced by M uniformly random genotypes."""
    np.random.seed(seed)
    L = asz.shape[0]
    g = np.zeros(L, np.int64)
    cnt = np.zeros(1 << nbits, np.int64)
    for m in range(M):
        for i in range(L):
            g[i] = np.random.randint(asz[i])
        cnt[table_of(enc, size, g, feat)] += 1
    return cnt


@njit(cache=True)
def enumerate_prog_tables(L, first, feat, nf, full, dflt, nins):
    """exact table counts over all programs of length L whose first instruction is `first`."""
    cnt = np.zeros(full + 1, np.int64)
    g = np.zeros(L, np.int64)
    g[0] = first
    total = 1
    for i in range(L - 1):
        total *= nins
    for idx in range(total):
        r = idx
        for i in range(1, L):
            g[i] = r % nins
            r //= nins
        cnt[prog_gen(g, L, feat, nf, full, dflt)] += 1
    return cnt


@njit(cache=True)
def run_chain2(enc, size, asz, g0, nsteps, seed, N, beta, rule, R, S, T, P, eps, delta,
               mh, mh_keys, mh_logp, mh_floor, rec_every, cap, feat, burn, mode, nwin, hD, hC):
    """Small-mutation-limit chain on genotypes; one mutation proposal per unit time.
    mode 0: point mutation (same random-number stream as v1 run_chain);
    mode 1: house-of-cards mutation (the mutant is a fresh uniformly random genotype, so mutant
            phenotypes are drawn exactly from the encoding prior P_E, independently of the resident).
    Extra outputs: window sums of resident cooperation (nwin windows after burn-in) and occupancy
    bookkeeping for the phenotypes with hashes hD (ALLD) and hC (ALLC):
    occ[k] = [time, entries, accepted neutral moves, exits to c>=0.5, exits to c<0.5]."""
    np.random.seed(seed)
    L = asz.shape[0]
    g = g0.copy()
    gm = g0.copy()
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    reg = Dict.empty(key_type=types.int64, value_type=types.int64)
    R_n = np.zeros(cap, np.int64)
    R_o = np.zeros((cap, NMAX), np.int8)
    R_t = np.zeros((cap, NMAX, 4), np.int8)
    R_h = np.zeros(cap, np.int64)
    selfpay = np.full(cap, np.nan)
    selfcoop = np.full(cap, np.nan)
    tin = np.zeros(cap)
    rcache = Dict.empty(key_type=types.int64, value_type=types.float64)
    coll = np.zeros(1, np.int64)
    nreg = 0
    tol = 1e-10
    oa = np.zeros(NMAX, np.int64)
    ta = np.zeros((NMAX, 4), np.int64)
    ob = np.zeros(NMAX, np.int64)
    tb = np.zeros((NMAX, 4), np.int64)
    maxlog = 6000000
    log_t = np.zeros(maxlog, np.int64)
    log_p = np.zeros(maxlog, np.int64)
    nrec = nsteps // rec_every + 1
    rec_g = np.zeros((nrec, L), np.int16)
    rec_p = np.zeros(nrec, np.int64)
    nr = 0
    win = np.zeros(nwin)
    wlen = (nsteps - burn) // nwin
    occ = np.zeros((2, 5))
    n_prop = 0
    n_neutral_prop = 0
    n_acc = 0
    n_acc_neutral = 0
    overflow = 0
    n = compile_raw2(enc, size, g, feat, out, tr)
    nc = minimise(n, out, tr, cout, ctr)
    hs = mhash(nc, cout, ctr)
    cur, nreg = _register_chk(nc, cout, ctr, hs, reg, R_n, R_o, R_t, R_h, nreg, coll)
    log_t[0] = 0
    log_p[0] = cur
    nlog = 1
    coop_sum = 0.0
    pay_sum = 0.0
    tot = 0.0
    for step in range(nsteps):
        if np.isnan(selfpay[cur]):
            na = _load(R_n, R_o, R_t, cur, oa, ta)
            u1, u2, c1, c2 = play(na, oa, ta, na, oa, ta, eps, delta, R, S, T, P, tol)
            selfpay[cur] = u1
            selfcoop[cur] = 0.5 * (c1 + c2)
        kcur = -1
        if R_h[cur] == hD:
            kcur = 0
        elif R_h[cur] == hC:
            kcur = 1
        if step >= burn:
            coop_sum += selfcoop[cur]
            pay_sum += selfpay[cur]
            tot += 1.0
            tin[cur] += 1.0
            wi = (step - burn) // wlen
            if wi < nwin:
                win[wi] += selfcoop[cur]
            if kcur >= 0:
                occ[kcur, 0] += 1.0
        if step % rec_every == 0 and nr < nrec:
            for i in range(L):
                rec_g[nr, i] = g[i]
            rec_p[nr] = cur
            nr += 1
        if mode == 0:
            i = np.random.randint(L)
            old = g[i]
            nv = np.random.randint(asz[i] - 1)
            if nv >= old:
                nv += 1
            g[i] = nv
        else:
            for j in range(L):
                gm[j] = g[j]
            for j in range(L):
                g[j] = np.random.randint(asz[j])
        n = compile_raw2(enc, size, g, feat, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        hs = mhash(nc, cout, ctr)
        mp, nreg2 = _register_chk(nc, cout, ctr, hs, reg, R_n, R_o, R_t, R_h, nreg, coll)
        if mp < 0:
            overflow = 1
            if mode == 0:
                g[i] = old
            else:
                for j in range(L):
                    g[j] = gm[j]
            break
        nreg = nreg2
        n_prop += 1
        if mp == cur:
            rho = 1.0 / N
            n_neutral_prop += 1
        else:
            key = cur * 4194304 + mp
            rho = -1.0
            if key in rcache:
                rho = rcache[key]
            if rho < 0.0:
                if np.isnan(selfpay[mp]):
                    nb_ = _load(R_n, R_o, R_t, mp, ob, tb)
                    u1, u2, c1, c2 = play(nb_, ob, tb, nb_, ob, tb, eps, delta, R, S, T, P, tol)
                    selfpay[mp] = u1
                    selfcoop[mp] = 0.5 * (c1 + c2)
                na = _load(R_n, R_o, R_t, cur, oa, ta)
                nb_ = _load(R_n, R_o, R_t, mp, ob, tb)
                uAB, uBA, c1, c2 = play(na, oa, ta, nb_, ob, tb, eps, delta, R, S, T, P, tol)
                rho = fixprob(selfpay[cur], uAB, uBA, selfpay[mp], N, beta, rule)
                rho2 = fixprob(selfpay[mp], uBA, uAB, selfpay[cur], N, beta, rule)
                if len(rcache) > 6000000:
                    rcache.clear()
                rcache[key] = rho
                rcache[mp * 4194304 + cur] = rho2
            if mh:
                h1 = R_h[cur]
                h2 = R_h[mp]
                k1 = np.searchsorted(mh_keys, h1)
                lp1 = mh_floor
                if k1 < mh_keys.shape[0] and mh_keys[k1] == h1:
                    lp1 = mh_logp[k1]
                k2 = np.searchsorted(mh_keys, h2)
                lp2 = mh_floor
                if k2 < mh_keys.shape[0] and mh_keys[k2] == h2:
                    lp2 = mh_logp[k2]
                a = np.exp(lp1 - lp2)
                if a < 1.0:
                    rho *= a
        if np.random.random() < rho:
            n_acc += 1
            if mp == cur:
                n_acc_neutral += 1
                if kcur >= 0 and step >= burn:
                    occ[kcur, 2] += 1.0
            else:
                if kcur >= 0 and step >= burn:
                    if np.isnan(selfcoop[mp]):
                        nb_ = _load(R_n, R_o, R_t, mp, ob, tb)
                        u1, u2, c1, c2 = play(nb_, ob, tb, nb_, ob, tb, eps, delta, R, S, T, P, tol)
                        selfpay[mp] = u1
                        selfcoop[mp] = 0.5 * (c1 + c2)
                    if selfcoop[mp] >= 0.5:
                        occ[kcur, 3] += 1.0
                    else:
                        occ[kcur, 4] += 1.0
                knew = -1
                if R_h[mp] == hD:
                    knew = 0
                elif R_h[mp] == hC:
                    knew = 1
                if knew >= 0 and step >= burn:
                    occ[knew, 1] += 1.0
                cur = mp
                if nlog < maxlog:
                    log_t[nlog] = step + 1
                    log_p[nlog] = cur
                    nlog += 1
        else:
            if mode == 0:
                g[i] = old
            else:
                for j in range(L):
                    g[j] = gm[j]
    stats = np.array([tot, coop_sum, pay_sum, n_prop, n_neutral_prop, n_acc, n_acc_neutral,
                      nreg, overflow, nlog, nr, coll[0], wlen], dtype=np.float64)
    return (stats, R_n[:nreg].copy(), R_o[:nreg].copy(), R_t[:nreg].copy(), R_h[:nreg].copy(),
            selfpay[:nreg].copy(), selfcoop[:nreg].copy(), tin[:nreg].copy(),
            log_t[:nlog].copy(), log_p[:nlog].copy(), rec_g[:nr].copy(), rec_p[:nr].copy(), win, occ)


@njit(cache=True)
def nbhd2(enc, size, asz, g, feat, eps, delta, R, S, T, P, N, beta, rule, nD, oD, tD):
    """single-mutant neighbourhood of genotype g, weighted by proposal probability.
    returns [neutral, spoiler, spoiler_any, invader, near_neutral, deleterious, other,
             rate_to_coop, rate_to_def, V(A), c(A), nonneutral]
    neutral: same phenotype; for phenotype-changing mutants B:
      spoiler: N rho(A,B) >= 1/2, c(B) >= 0.5, V(B) >= 1 and V(A) < 1
      spoiler_any: N rho(A,B) >= 1/2, c(B) >= 0.5, V(B) >= 1 (no condition on A)
      invader: N rho(A,B) >= 1/2 and c(B) < 0.5
      near_neutral: 0.8 <= N rho(A,B) <= 1.25 (overlaps the classes above)
      deleterious: N rho(A,B) < 1/2
    rate_to_*: per-proposal probability that a mutant of that kind arises and fixes."""
    L = asz.shape[0]
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    oA = np.zeros(NMAX, np.int64)
    tA = np.zeros((NMAX, 4), np.int64)
    oB = np.zeros(NMAX, np.int64)
    tB = np.zeros((NMAX, 4), np.int64)
    gg = g.copy()
    n = compile_raw2(enc, size, gg, feat, out, tr)
    nA = minimise(n, out, tr, oA, tA)
    hA = mhash(nA, oA, tA)
    u1, u2, c1, c2 = play(nA, oA, tA, nA, oA, tA, eps, delta, R, S, T, P, 1e-10)
    cA = 0.5 * (c1 + c2)
    vA_, x, cD, cA_ = rho_pair(nA, oA, tA, nD, oD, tD, eps, delta, R, S, T, P, N, beta, rule)
    VA = N * vA_
    res = np.zeros(12)
    res[9] = VA
    res[10] = cA
    for i in range(L):
        old = gg[i]
        for v in range(asz[i]):
            if v == old:
                continue
            w = 1.0 / L / (asz[i] - 1)
            gg[i] = v
            n = compile_raw2(enc, size, gg, feat, out, tr)
            nB = minimise(n, out, tr, oB, tB)
            hB = mhash(nB, oB, tB)
            if hB == hA:
                res[0] += w
            else:
                res[11] += w
                rAB, rBA, cB, cA2 = rho_pair(nA, oA, tA, nB, oB, tB, eps, delta, R, S, T, P, N, beta, rule)
                vB, x2, cD2, cB2 = rho_pair(nB, oB, tB, nD, oD, tD, eps, delta, R, S, T, P, N, beta, rule)
                VB = N * vB
                xr = N * rAB
                if xr >= 0.5:
                    if cB < 0.5:
                        res[3] += w
                    elif VB >= 1.0:
                        res[2] += w
                        if VA < 1.0:
                            res[1] += w
                        else:
                            res[6] += w
                    else:
                        res[6] += w
                else:
                    res[5] += w
                if xr >= 0.8 and xr <= 1.25:
                    res[4] += w
                if cB >= 0.5:
                    res[7] += w * rAB
                else:
                    res[8] += w * rAB
        gg[i] = old
    return res


@njit(cache=True)
def sample_network2(enc, size, asz, target, K, maxtries, seed, feat):
    np.random.seed(seed)
    L = asz.shape[0]
    outg = np.zeros((K, L), np.int64)
    g = np.zeros(L, np.int64)
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    k = 0
    tries = 0
    while k < K and tries < maxtries:
        tries += 1
        for i in range(L):
            g[i] = np.random.randint(asz[i])
        n = compile_raw2(enc, size, g, feat, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        if mhash(nc, cout, ctr) == target:
            for i in range(L):
                outg[k, i] = g[i]
            k += 1
    return outg[:k].copy(), tries

@njit(cache=True)
def ibm_run(enc, size, asz, g0, N, steps, burn, mu, beta, R, S, T, P, eps, delta, seed, feat, cap, sample_every):
    """Individual-based pairwise-comparison (imitation) process with point mutation at rate mu.
    Each step: a random focal individual mutates with probability mu; otherwise it compares itself
    with a random other individual and copies its genotype with probability
    1/(1+exp(-beta*(pi_model - pi_focal))). Payoffs are averages over games with the N-1 others.
    Returns time-averaged population cooperation (average over ordered pairs of distinct individuals),
    time share of monomorphic states, mean number of distinct phenotypes, and mutation count."""
    np.random.seed(seed)
    L = asz.shape[0]
    G = np.zeros((N, L), np.int64)
    for i in range(N):
        for j in range(L):
            G[i, j] = g0[j]
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    reg = Dict.empty(key_type=types.int64, value_type=types.int64)
    R_n = np.zeros(cap, np.int64)
    R_o = np.zeros((cap, NMAX), np.int8)
    R_t = np.zeros((cap, NMAX, 4), np.int8)
    R_h = np.zeros(cap, np.int64)
    coll = np.zeros(1, np.int64)
    pu = Dict.empty(key_type=types.int64, value_type=types.float64)
    pc = Dict.empty(key_type=types.int64, value_type=types.float64)
    oa = np.zeros(NMAX, np.int64)
    ta = np.zeros((NMAX, 4), np.int64)
    ob = np.zeros(NMAX, np.int64)
    tb = np.zeros((NMAX, 4), np.int64)
    nreg = 0
    n = compile_raw2(enc, size, G[0], feat, out, tr)
    nc = minimise(n, out, tr, cout, ctr)
    p0, nreg = _register_chk(nc, cout, ctr, mhash(nc, cout, ctr), reg, R_n, R_o, R_t, R_h, nreg, coll)
    pid = np.full(N, p0, np.int64)
    cnt = np.zeros(cap, np.int64)
    cnt[p0] = N
    present = np.zeros(N + 1, np.int64)
    pos = np.full(cap, -1, np.int64)
    present[0] = p0
    pos[p0] = 0
    K = 1
    Up = np.zeros(cap)
    Uc = np.zeros(cap)
    selfu = np.zeros(cap)
    selfc = np.zeros(cap)
    # helper closures are not allowed in numba; inline pair evaluation via cache
    na = _load(R_n, R_o, R_t, p0, oa, ta)
    u1, u2, c1, c2 = play(na, oa, ta, na, oa, ta, eps, delta, R, S, T, P, 1e-10)
    pu[p0 * 4194304 + p0] = u1
    pc[p0 * 4194304 + p0] = c1
    selfu[p0] = u1
    selfc[p0] = c1
    Up[p0] = N * u1
    Uc[p0] = N * c1
    coop_acc = 0.0
    mono_acc = 0.0
    k_acc = 0.0
    nsamp = 0.0
    nmut = 0
    overflow = 0
    for step in range(steps):
        i = np.random.randint(N)
        if np.random.random() < mu:
            # point mutation of individual i
            loc = np.random.randint(L)
            old = G[i, loc]
            nv = np.random.randint(asz[loc] - 1)
            if nv >= old:
                nv += 1
            G[i, loc] = nv
            nmut += 1
            n = compile_raw2(enc, size, G[i], feat, out, tr)
            nc = minimise(n, out, tr, cout, ctr)
            newp, nreg2 = _register_chk(nc, cout, ctr, mhash(nc, cout, ctr), reg, R_n, R_o, R_t, R_h, nreg, coll)
            if newp < 0:
                overflow = 1
                break
            nreg = nreg2
        else:
            j = np.random.randint(N - 1)
            if j >= i:
                j += 1
            a = pid[i]
            b = pid[j]
            newp = -1
            if a == b:
                # equal payoffs: imitation probability 1/2 (neutral genotype drift)
                if np.random.random() < 0.5:
                    for k2 in range(L):
                        G[i, k2] = G[j, k2]
            else:
                pia = (Up[a] - selfu[a]) / (N - 1)
                pib = (Up[b] - selfu[b]) / (N - 1)
                if np.random.random() < 1.0 / (1.0 + np.exp(-beta * (pib - pia))):
                    for k2 in range(L):
                        G[i, k2] = G[j, k2]
                    newp = b
        if newp >= 0 and newp != pid[i]:
            a = pid[i]
            # register newp in present set if needed
            if cnt[newp] == 0:
                # compute its pair values with all present phenotypes
                nb_ = _load(R_n, R_o, R_t, newp, ob, tb)
                key = newp * 4194304 + newp
                if key not in pu:
                    u1, u2, c1, c2 = play(nb_, ob, tb, nb_, ob, tb, eps, delta, R, S, T, P, 1e-10)
                    pu[key] = u1
                    pc[key] = c1
                selfu[newp] = pu[key]
                selfc[newp] = pc[key]
                upn = 0.0
                ucn = 0.0
                for q in range(K):
                    x = present[q]
                    k1 = newp * 4194304 + x
                    if k1 not in pu:
                        na = _load(R_n, R_o, R_t, x, oa, ta)
                        uAB, uBA, cAB, cBA = play(nb_, ob, tb, na, oa, ta, eps, delta, R, S, T, P, 1e-10)
                        pu[k1] = uAB
                        pc[k1] = cAB
                        pu[x * 4194304 + newp] = uBA
                        pc[x * 4194304 + newp] = cBA
                    upn += cnt[x] * pu[k1]
                    ucn += cnt[x] * pc[k1]
                Up[newp] = upn
                Uc[newp] = ucn
                present[K] = newp
                pos[newp] = K
                K += 1
            # count changes: a -1, newp +1
            for q in range(K):
                x = present[q]
                Up[x] += pu[x * 4194304 + newp] - pu[x * 4194304 + a]
                Uc[x] += pc[x * 4194304 + newp] - pc[x * 4194304 + a]
            cnt[a] -= 1
            cnt[newp] += 1
            pid[i] = newp
            if cnt[a] == 0:
                q = pos[a]
                last = present[K - 1]
                present[q] = last
                pos[last] = q
                pos[a] = -1
                K -= 1
                if len(pu) > 4000000:
                    pass
        if step >= burn and (step % sample_every) == 0:
            cs = 0.0
            for q in range(K):
                x = present[q]
                cs += cnt[x] * (Uc[x] - selfc[x])
            coop_acc += cs / (N * (N - 1.0))
            if K == 1:
                mono_acc += 1.0
            k_acc += K
            nsamp += 1.0
    return coop_acc / nsamp, mono_acc / nsamp, k_acc / nsamp, nmut, overflow, coll[0], nreg
