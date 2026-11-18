"""
sb_core.py -- genotype-phenotype maps for repeated-game strategies, phenotype
canonicalisation (minimal Moore machines), exact repeated-game payoffs, and a
small-mutation-limit (SML) Markov chain on genotypes.

Conventions
-----------
Action 1 = cooperate (C), 0 = defect (D).
Outcome seen by a player, o = 2*own + opp  (0 = DD, 1 = own D/opp C, 2 = own C/opp D, 3 = CC).
A phenotype is the behaviour of a strategy on *all* finite histories (including
histories reachable only through execution errors), represented by the minimal
Moore machine over the 4-letter outcome alphabet, canonically relabelled by
breadth-first search from the start state.

Encodings (enc id):
 0  TAB1  memory-one lookup table            genes: 5 binary
 1  TAB2  memory-two window lookup table     genes: 21 binary
 2  CIRC  feed-forward Boolean circuit       genes: 3G (gate type in 4, two inputs)
 3  PROG  stack-language program             genes: L (12 instructions)
 4  AUTO  Moore automaton, k states          genes: k outputs (2) + 4k transitions (k)
 5  RNN   binary recurrent threshold network genes: H^2 + 5H + 1 (weights -2..2) + H (h0 bits)
Encodings 1-3 share one phenotype set (the 2^21 memory-two window strategies).
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

NMAX = 32
ENC_NAMES = {0: "TAB1", 1: "TAB2", 2: "CIRC", 3: "PROG", 4: "AUTO", 5: "RNN"}
FULL21 = (1 << 21) - 1


def gene_alphabet(enc, size):
    if enc == 0:
        return np.full(5, 2, np.int64)
    if enc == 1:
        return np.full(21, 2, np.int64)
    if enc == 2:
        a = []
        for j in range(size):
            a += [4, 6 + j, 6 + j]
        return np.array(a, np.int64)
    if enc == 3:
        return np.full(size, 12, np.int64)
    if enc == 4:
        k = size
        return np.array([2] * k + [k] * (4 * k), np.int64)
    if enc == 5:
        H = size
        return np.array([5] * (H * H + 2 * H + H + H + 1) + [2] * H, np.int64)
    raise ValueError(enc)


def _feature_masks():
    F = np.zeros(6, np.int64)   # a1, b1, a2, b2, h1, h2
    for k in range(21):
        if k == 0:
            f = (0, 0, 0, 0, 1, 0)
        elif k <= 4:
            o1 = k - 1
            f = (o1 >> 1, o1 & 1, 0, 0, 0, 1)
        else:
            idx = k - 5
            o2, o1 = idx >> 2, idx & 3
            f = (o1 >> 1, o1 & 1, o2 >> 1, o2 & 1, 0, 0)
        for i in range(6):
            if f[i]:
                F[i] |= (1 << k)
    return F


FEAT = _feature_masks()


@njit(cache=True)
def circ_table(g, G, feat):
    vals = np.zeros(6 + G, np.int64)
    for i in range(6):
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
            v = (~(x & y)) & FULL21
        else:
            v = (~(x | y)) & FULL21
        vals[6 + j] = v
    return vals[6 + G - 1]


@njit(cache=True)
def prog_table(g, L, feat):
    st = np.zeros(L + 1, np.int64)
    sp = 0
    for i in range(L):
        op = g[i]
        if op < 6:
            st[sp] = feat[op]
            sp += 1
        elif op == 6:
            if sp >= 1:
                st[sp - 1] = (~st[sp - 1]) & FULL21
        elif op <= 9:
            if sp >= 2:
                x = st[sp - 1]
                y = st[sp - 2]
                if op == 7:
                    v = x & y
                elif op == 8:
                    v = x | y
                else:
                    v = x ^ y
                st[sp - 2] = v
                sp -= 1
        elif op == 10:
            if sp >= 1:
                st[sp] = st[sp - 1]
                sp += 1
        else:
            if sp >= 2:
                x = st[sp - 1]
                st[sp - 1] = st[sp - 2]
                st[sp - 2] = x
    if sp == 0:
        return 0
    return st[sp - 1]


@njit(cache=True)
def table2_machine(tab, out, tr):
    for k in range(21):
        out[k] = (tab >> k) & 1
    for o in range(4):
        tr[0, o] = 1 + o
    for o1 in range(4):
        for o in range(4):
            tr[1 + o1, o] = 5 + 4 * o1 + o
    for o2 in range(4):
        for o1 in range(4):
            for o in range(4):
                tr[5 + 4 * o2 + o1, o] = 5 + 4 * o1 + o
    return 21


@njit(cache=True)
def compile_raw(enc, size, g, feat, out, tr):
    if enc == 0:
        out[0] = g[0]
        for o in range(4):
            tr[0, o] = 1 + o
        for s in range(4):
            out[1 + s] = g[1 + s]
            for o in range(4):
                tr[1 + s, o] = 1 + o
        return 5
    if enc == 1:
        tab = 0
        for k in range(21):
            if g[k]:
                tab |= (1 << k)
        return table2_machine(tab, out, tr)
    if enc == 2:
        return table2_machine(circ_table(g, size, feat), out, tr)
    if enc == 3:
        return table2_machine(prog_table(g, size, feat), out, tr)
    if enc == 4:
        k = size
        for s in range(k):
            out[s] = g[s]
            for o in range(4):
                tr[s, o] = g[k + 4 * s + o]
        return k
    H = size
    off_W = 0
    off_U = H * H
    off_c = off_U + 2 * H
    off_v = off_c + H
    off_d = off_v + H
    off_h = off_d + 1
    ns = 1 << H
    hb = np.zeros(H, np.int64)
    for s in range(ns):
        for i in range(H):
            hb[i] = 2 * ((s >> i) & 1) - 1
        z = g[off_d] - 2
        for i in range(H):
            z += (g[off_v + i] - 2) * hb[i]
        out[s] = 1 if z > 0 else 0
        for o in range(4):
            xa = 2 * (o >> 1) - 1
            xb = 2 * (o & 1) - 1
            nxt = 0
            for i in range(H):
                zi = g[off_c + i] - 2
                for j in range(H):
                    zi += (g[off_W + i * H + j] - 2) * hb[j]
                zi += (g[off_U + 2 * i] - 2) * xa + (g[off_U + 2 * i + 1] - 2) * xb
                if zi > 0:
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
def minimise(n, out, tr, cout, ctr):
    reach = np.full(n, -1, np.int64)
    order = np.zeros(n, np.int64)
    reach[0] = 0
    order[0] = 0
    m = 1
    h = 0
    while h < m:
        s = order[h]
        h += 1
        for o in range(4):
            t = tr[s, o]
            if reach[t] < 0:
                reach[t] = m
                order[m] = t
                m += 1
    lo = np.zeros(m, np.int64)
    lt = np.zeros((m, 4), np.int64)
    for i in range(m):
        s = order[i]
        lo[i] = out[s]
        for o in range(4):
            lt[i, o] = reach[tr[s, o]]
    blk = np.zeros(m, np.int64)
    nb = 1
    for i in range(m):
        if lo[i] != lo[0]:
            blk[i] = 1
            nb = 2
    sig = np.zeros((m, 5), np.int64)
    newb = np.zeros(m, np.int64)
    while True:
        for i in range(m):
            sig[i, 0] = blk[i]
            for o in range(4):
                sig[i, 1 + o] = blk[lt[i, o]]
        cnt = 0
        for i in range(m):
            found = -1
            for j in range(i):
                same = True
                for c in range(5):
                    if sig[i, c] != sig[j, c]:
                        same = False
                        break
                if same:
                    found = newb[j]
                    break
            if found < 0:
                newb[i] = cnt
                cnt += 1
            else:
                newb[i] = found
        for i in range(m):
            blk[i] = newb[i]
        if cnt == nb:
            break
        nb = cnt
    qo = np.zeros(nb, np.int64)
    qt = np.zeros((nb, 4), np.int64)
    for i in range(m):
        b = blk[i]
        qo[b] = lo[i]
        for o in range(4):
            qt[b, o] = blk[lt[i, o]]
    lab = np.full(nb, -1, np.int64)
    q = np.zeros(nb, np.int64)
    lab[blk[0]] = 0
    q[0] = blk[0]
    mm = 1
    h = 0
    while h < mm:
        b = q[h]
        h += 1
        for o in range(4):
            t = qt[b, o]
            if lab[t] < 0:
                lab[t] = mm
                q[mm] = t
                mm += 1
    for b in range(nb):
        L = lab[b]
        cout[L] = qo[b]
        for o in range(4):
            ctr[L, o] = lab[qt[b, o]]
    return nb


@njit(cache=True)
def mhash(n, cout, ctr):
    h = np.uint64(14695981039346656037)
    p = np.uint64(1099511628211)
    h = (h ^ np.uint64(n + 1)) * p
    for s in range(n):
        h = (h ^ np.uint64(cout[s] + 7)) * p
        for o in range(4):
            h = (h ^ np.uint64(ctr[s, o] + 11)) * p
    return np.int64(h & np.uint64(0x7FFFFFFFFFFFFFFF))


@njit(cache=True)
def play(nA, oA, tA, nB, oB, tB, eps, delta, R, S, T, P, tol):
    pay = np.array([[P, T], [S, R]])
    ns = nA * nB
    x = np.zeros(ns)
    y = np.zeros(ns)
    x[0] = 1.0
    uA = 0.0
    uB = 0.0
    cA = 0.0
    cB = 0.0
    w = 1.0 - delta
    while w > tol:
        for k in range(ns):
            y[k] = 0.0
        for k in range(ns):
            mk = x[k]
            if mk == 0.0:
                continue
            i = k // nB
            j = k - i * nB
            a = oA[i]
            b = oB[j]
            for ra in range(2):
                pa = (1.0 - eps) if ra == a else eps
                if pa == 0.0:
                    continue
                for rb in range(2):
                    pb = (1.0 - eps) if rb == b else eps
                    if pb == 0.0:
                        continue
                    pr = mk * pa * pb
                    uA += w * pr * pay[ra, rb]
                    uB += w * pr * pay[rb, ra]
                    cA += w * pr * ra
                    cB += w * pr * rb
                    ni = tA[i, 2 * ra + rb]
                    nj = tB[j, 2 * rb + ra]
                    y[ni * nB + nj] += pr
        for k in range(ns):
            x[k] = y[k]
        w *= delta
    tot = 1.0 - w / (1.0 - delta)
    return uA / tot, uB / tot, cA / tot, cB / tot


@njit(cache=True)
def fixprob(uAA, uAB, uBA, uBB, N, beta, rule):
    """fixation probability of one B mutant in an A population (N individuals).
    rule 0: pairwise comparison (Fermi), intensity beta;
    rule 1: Moran birth-death, fitness 1 + beta*payoff."""
    Lk = 0.0
    mx = 0.0
    acc = np.zeros(N)
    for j in range(1, N):
        pA = ((N - j - 1) * uAA + j * uAB) / (N - 1)
        pB = ((j - 1) * uBB + (N - j) * uBA) / (N - 1)
        if rule == 0:
            Lk += -beta * (pB - pA)
        else:
            Lk += np.log((1.0 + beta * pA) / (1.0 + beta * pB))
        acc[j] = Lk
        if Lk > mx:
            mx = Lk
    s = 0.0
    for j in range(N):
        s += np.exp(acc[j] - mx)
    return np.exp(-mx) / s


@njit(cache=True)
def _register(n, cout, ctr, hsh, reg, R_n, R_o, R_t, R_h, nreg):
    if hsh in reg:
        return reg[hsh], nreg
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
def _load(R_n, R_o, R_t, p, o, t):
    n = R_n[p]
    for s in range(n):
        o[s] = R_o[p, s]
        for k in range(4):
            t[s, k] = R_t[p, s, k]
    return n


@njit(cache=True)
def run_chain(enc, size, asz, g0, nsteps, seed, N, beta, rule, R, S, T, P, eps, delta,
              mh, mh_keys, mh_logp, mh_floor, rec_every, cap, feat, burn):
    """Small-mutation-limit chain on genotypes; one point-mutation proposal per unit time."""
    np.random.seed(seed)
    L = asz.shape[0]
    g = g0.copy()
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
    nreg = 0
    tol = 1e-10
    oa = np.zeros(NMAX, np.int64)
    ta = np.zeros((NMAX, 4), np.int64)
    ob = np.zeros(NMAX, np.int64)
    tb = np.zeros((NMAX, 4), np.int64)
    maxlog = 3000000
    log_t = np.zeros(maxlog, np.int64)
    log_p = np.zeros(maxlog, np.int64)
    nrec = nsteps // rec_every + 1
    rec_g = np.zeros((nrec, L), np.int16)
    rec_p = np.zeros(nrec, np.int64)
    nr = 0
    n_prop = 0
    n_neutral_prop = 0
    n_acc = 0
    n_acc_neutral = 0
    overflow = 0
    n = compile_raw(enc, size, g, feat, out, tr)
    nc = minimise(n, out, tr, cout, ctr)
    hs = mhash(nc, cout, ctr)
    cur, nreg = _register(nc, cout, ctr, hs, reg, R_n, R_o, R_t, R_h, nreg)
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
        if step >= burn:
            coop_sum += selfcoop[cur]
            pay_sum += selfpay[cur]
            tot += 1.0
            tin[cur] += 1.0
        if step % rec_every == 0 and nr < nrec:
            for i in range(L):
                rec_g[nr, i] = g[i]
            rec_p[nr] = cur
            nr += 1
        i = np.random.randint(L)
        old = g[i]
        nv = np.random.randint(asz[i] - 1)
        if nv >= old:
            nv += 1
        g[i] = nv
        n = compile_raw(enc, size, g, feat, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        hs = mhash(nc, cout, ctr)
        mp, nreg2 = _register(nc, cout, ctr, hs, reg, R_n, R_o, R_t, R_h, nreg)
        if mp < 0:
            overflow = 1
            g[i] = old
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
            else:
                cur = mp
                if nlog < maxlog:
                    log_t[nlog] = step + 1
                    log_p[nlog] = cur
                    nlog += 1
        else:
            g[i] = old
    stats = np.array([tot, coop_sum, pay_sum, n_prop, n_neutral_prop, n_acc, n_acc_neutral,
                      nreg, overflow, nlog, nr], dtype=np.float64)
    return (stats, R_n[:nreg].copy(), R_o[:nreg].copy(), R_t[:nreg].copy(), R_h[:nreg].copy(),
            selfpay[:nreg].copy(), selfcoop[:nreg].copy(), tin[:nreg].copy(),
            log_t[:nlog].copy(), log_p[:nlog].copy(), rec_g[:nr].copy(), rec_p[:nr].copy())


@njit(cache=True)
def sample_map(enc, size, asz, M, seed, feat, cap):
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
    nreg = 0
    over = 0
    for m in range(M):
        for i in range(L):
            g[i] = np.random.randint(asz[i])
        n = compile_raw(enc, size, g, feat, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        hs = mhash(nc, cout, ctr)
        pid, nreg2 = _register(nc, cout, ctr, hs, reg, R_n, R_o, R_t, R_h, nreg)
        if pid < 0:
            over += 1
            continue
        nreg = nreg2
        cnt[pid] += 1
    return cnt[:nreg].copy(), R_n[:nreg].copy(), R_o[:nreg].copy(), R_t[:nreg].copy(), R_h[:nreg].copy(), over


@njit(cache=True)
def phen_of(enc, size, g, feat):
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    n = compile_raw(enc, size, g, feat, out, tr)
    nc = minimise(n, out, tr, cout, ctr)
    return nc, cout[:nc].copy(), ctr[:nc].copy(), mhash(nc, cout, ctr)


@njit(cache=True)
def enumerate_table2(feat):
    M = 1 << 21
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    ns = np.zeros(M, np.int8)
    hs = np.zeros(M, np.int64)
    for t in range(M):
        n = table2_machine(t, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        ns[t] = nc
        hs[t] = mhash(nc, cout, ctr)
    return ns, hs


@njit(cache=True)
def fingerprint(n, o, t, depth):
    tot = 0
    p = 1
    for d in range(depth + 1):
        tot += p
        p *= 4
    fp = np.zeros(tot, np.int8)
    cur = np.zeros(tot, np.int64)
    cur[0] = 0
    fp[0] = o[0]
    start = 0
    w = 1
    for d in range(depth):
        k = start + w
        for i in range(start, start + w):
            s = cur[i]
            for oo in range(4):
                ns = t[s, oo]
                cur[k] = ns
                fp[k] = o[ns]
                k += 1
        start = start + w
        w *= 4
    return fp


@njit(cache=True)
def lz76_words(s):
    n = s.shape[0]
    if n <= 1:
        return n
    c = 1
    l = 1
    i = 0
    k = 1
    kmax = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1
    return c


@njit(cache=True)
def k_lz(s):
    n = s.shape[0]
    allsame = True
    for i in range(1, n):
        if s[i] != s[0]:
            allsame = False
            break
    if allsame:
        return np.log2(n)
    r = s[::-1].copy()
    return np.log2(n) * (lz76_words(s) + lz76_words(r)) / 2.0


def k_desc(n):
    n = np.asarray(n, float)
    return n * (1.0 + 4.0 * np.log2(np.maximum(n, 1.0)))


@njit(cache=True)
def complexities(R_n, R_o, R_t, depth):
    m = R_n.shape[0]
    klz = np.zeros(m)
    o = np.zeros(NMAX, np.int64)
    t = np.zeros((NMAX, 4), np.int64)
    for p in range(m):
        n = _load(R_n, R_o, R_t, p, o, t)
        fp = fingerprint(n, o, t, depth)
        klz[p] = k_lz(fp)
    return klz


def memory_one(p0, pCC, pCD, pDC, pDD):
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    out[0] = p0
    resp = {3: pCC, 2: pCD, 1: pDC, 0: pDD}
    for s in range(5):
        if s > 0:
            out[s] = resp[s - 1]
        for o in range(4):
            tr[s, o] = 1 + o
    return 5, out, tr


def named_machines():
    M = {}
    M["ALLD"] = memory_one(0, 0, 0, 0, 0)
    M["ALLC"] = memory_one(1, 1, 1, 1, 1)
    M["TFT"] = memory_one(1, 1, 0, 1, 0)
    M["STFT"] = memory_one(0, 1, 0, 1, 0)
    M["WSLS"] = memory_one(1, 1, 0, 0, 1)
    M["ALT"] = memory_one(1, 0, 0, 1, 1)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1] = 1, 0
    for o in range(4):
        tr[0, o] = 0 if (o & 1) else 1
        tr[1, o] = 1
    M["GRIM"] = (2, out, tr)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2] = 1, 1, 0
    for s in range(3):
        for o in range(4):
            tr[s, o] = 0 if (o & 1) else (1 if s == 0 else 2)
    M["TF2T"] = (3, out, tr)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2] = 1, 0, 0
    for o in range(4):
        tr[0, o] = 0 if (o & 1) else 1
        tr[1, o] = 0 if (o & 1) else 1
        tr[2, o] = 0
    # 2TFT: after an opponent D, defect in the next two rounds (restart counting on new D)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2] = 1, 0, 0
    for s in range(3):
        for o in range(4):
            if not (o & 1):
                tr[s, o] = 1
            else:
                tr[s, o] = 2 if s == 1 else 0
    M["2TFT"] = (3, out, tr)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2] = 1, 0, 0
    for o in range(4):
        tr[0, o] = 0 if (o & 1) else 1
        tr[1, o] = 2
        tr[2, o] = 0
    M["T2"] = (3, out, tr)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2] = 1, 1, 0
    for o in range(4):
        tr[0, o] = 0 if o == 3 else 1
        tr[1, o] = 0 if o == 3 else 2
        tr[2, o] = 2
    M["GRIM2"] = (3, out, tr)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2], out[3] = 1, 1, 1, 0
    for o in range(4):
        tr[0, o] = 0 if o == 3 else 1
        tr[1, o] = 0 if o == 3 else 2
        tr[2, o] = 0 if o == 3 else 3
        tr[3, o] = 3
    M["GRIM3"] = (4, out, tr)
    out = np.zeros(NMAX, np.int64); tr = np.zeros((NMAX, 4), np.int64)
    out[0], out[1], out[2], out[3] = 1, 1, 1, 0
    for s in range(4):
        for o in range(4):
            tr[s, o] = 0 if (o & 1) else min(s + 1, 3)
    M["TF3T"] = (4, out, tr)
    canon = {}
    for k, (n, o, t) in M.items():
        co = np.zeros(NMAX, np.int64); ct = np.zeros((NMAX, 4), np.int64)
        nc = minimise(n, o, t, co, ct)
        canon[k] = (nc, co[:nc].copy(), ct[:nc].copy(), int(mhash(nc, co, ct)))
    return canon


@njit(cache=True)
def selfplay_all(R_n, R_o, R_t, eps, delta, R, S, T, P):
    m = R_n.shape[0]
    coop = np.zeros(m)
    pay = np.zeros(m)
    o = np.zeros(NMAX, np.int64)
    t = np.zeros((NMAX, 4), np.int64)
    for p in range(m):
        n = _load(R_n, R_o, R_t, p, o, t)
        u1, u2, c1, c2 = play(n, o, t, n, o, t, eps, delta, R, S, T, P, 1e-10)
        coop[p] = 0.5 * (c1 + c2)
        pay[p] = u1
    return coop, pay


@njit(cache=True)
def table2_stats(feat, eps, delta, R, S, T, P, depth):
    M = 1 << 21
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    cout = np.zeros(NMAX, np.int64)
    ctr = np.zeros((NMAX, 4), np.int64)
    ns = np.zeros(M, np.int64)
    hs = np.zeros(M, np.int64)
    klz = np.zeros(M)
    coop = np.zeros(M)
    for tab in range(M):
        n = table2_machine(tab, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        ns[tab] = nc
        hs[tab] = mhash(nc, cout, ctr)
        fp = fingerprint(nc, cout, ctr, depth)
        klz[tab] = k_lz(fp)
        u1, u2, c1, c2 = play(nc, cout, ctr, nc, cout, ctr, eps, delta, R, S, T, P, 1e-10)
        coop[tab] = 0.5 * (c1 + c2)
    return ns, hs, klz, coop


@njit(cache=True)
def rho_pair(nA, oA, tA, nB, oB, tB, eps, delta, R, S, T, P, N, beta, rule):
    """fixation probabilities B->A-population and A->B-population, plus self-play coop of B."""
    uAA, x1, cA1, cA2 = play(nA, oA, tA, nA, oA, tA, eps, delta, R, S, T, P, 1e-10)
    uBB, x2, cB1, cB2 = play(nB, oB, tB, nB, oB, tB, eps, delta, R, S, T, P, 1e-10)
    uAB, uBA, c1, c2 = play(nA, oA, tA, nB, oB, tB, eps, delta, R, S, T, P, 1e-10)
    rAB = fixprob(uAA, uAB, uBA, uBB, N, beta, rule)   # B invades A
    rBA = fixprob(uBB, uBA, uAB, uAA, N, beta, rule)   # A invades B
    return rAB, rBA, 0.5 * (cB1 + cB2), 0.5 * (cA1 + cA2)


@njit(cache=True)
def nbhd_stats(enc, size, asz, g, feat, eps, delta, R, S, T, P, N, beta, rule, nD, oD, tD):
    """mutational neighbourhood of resident genotype g, weighted by proposal probability.
    returns [w_neutral, w_spoiler, w_invader, w_robust_nn, w_deleterious, w_other, V(A), c(A)]"""
    L = asz.shape[0]
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    oA = np.zeros(NMAX, np.int64)
    tA = np.zeros((NMAX, 4), np.int64)
    oB = np.zeros(NMAX, np.int64)
    tB = np.zeros((NMAX, 4), np.int64)
    gg = g.copy()
    n = compile_raw(enc, size, gg, feat, out, tr)
    nA = minimise(n, out, tr, oA, tA)
    hA = mhash(nA, oA, tA)
    vA_, x, cD, cA = rho_pair(nA, oA, tA, nD, oD, tD, eps, delta, R, S, T, P, N, beta, rule)
    VA = N * vA_
    res = np.zeros(8)
    res[6] = VA
    res[7] = cA
    for i in range(L):
        old = gg[i]
        for v in range(asz[i]):
            if v == old:
                continue
            w = 1.0 / L / (asz[i] - 1)
            gg[i] = v
            n = compile_raw(enc, size, gg, feat, out, tr)
            nB = minimise(n, out, tr, oB, tB)
            hB = mhash(nB, oB, tB)
            if hB == hA:
                res[0] += w
            else:
                rAB, rBA, cB, cA2 = rho_pair(nA, oA, tA, nB, oB, tB, eps, delta, R, S, T, P, N, beta, rule)
                vB, x2, cD2, cB2 = rho_pair(nB, oB, tB, nD, oD, tD, eps, delta, R, S, T, P, N, beta, rule)
                VB = N * vB
                if N * rAB >= 0.5:
                    if cB < 0.5:
                        res[2] += w
                    elif VB >= 1.0 and VA < 1.0:
                        res[1] += w
                    else:
                        res[3] += w
                elif N * rAB < 0.5:
                    res[4] += w
                else:
                    res[5] += w
        gg[i] = old
    return res


@njit(cache=True)
def nbhd_focal(enc, size, asz, g, feat, eps, delta, R, S, T, P, N, beta, rule):
    """neighbourhood of genotype g (any phenotype A), weighted by proposal probability.
    returns [w_neutral, w_adv_coop, w_adv_def, w_nearneutral, w_deleterious,
             rate_to_coop, rate_to_def, rate_total, c(A)]
    adv: N*rho >= 1; nearneutral: 0.5 <= N*rho < 1; coop: self-play cooperation >= 0.5;
    rate_*: per-proposal probability that the mutant arises and fixes (sum of w*rho)."""
    L = asz.shape[0]
    out = np.zeros(NMAX, np.int64)
    tr = np.zeros((NMAX, 4), np.int64)
    oA = np.zeros(NMAX, np.int64)
    tA = np.zeros((NMAX, 4), np.int64)
    oB = np.zeros(NMAX, np.int64)
    tB = np.zeros((NMAX, 4), np.int64)
    gg = g.copy()
    n = compile_raw(enc, size, gg, feat, out, tr)
    nA = minimise(n, out, tr, oA, tA)
    hA = mhash(nA, oA, tA)
    res = np.zeros(9)
    cA = -1.0
    for i in range(L):
        old = gg[i]
        for v in range(asz[i]):
            if v == old:
                continue
            w = 1.0 / L / (asz[i] - 1)
            gg[i] = v
            n = compile_raw(enc, size, gg, feat, out, tr)
            nB = minimise(n, out, tr, oB, tB)
            hB = mhash(nB, oB, tB)
            if hB == hA:
                res[0] += w
                res[7] += w / N
            else:
                rAB, rBA, cB, cA = rho_pair(nA, oA, tA, nB, oB, tB, eps, delta, R, S, T, P, N, beta, rule)
                x = N * rAB
                if x >= 1.0:
                    if cB >= 0.5:
                        res[1] += w
                    else:
                        res[2] += w
                elif x >= 0.5:
                    res[3] += w
                else:
                    res[4] += w
                if cB >= 0.5:
                    res[5] += w * rAB
                else:
                    res[6] += w * rAB
                res[7] += w * rAB
        gg[i] = old
    res[8] = cA
    return res


@njit(cache=True)
def sample_network(enc, size, asz, target, K, maxtries, seed, feat):
    """uniform sample of up to K genotypes from the neutral set of phenotype hash `target`."""
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
        n = compile_raw(enc, size, g, feat, out, tr)
        nc = minimise(n, out, tr, cout, ctr)
        if mhash(nc, cout, ctr) == target:
            for i in range(L):
                outg[k, i] = g[i]
            k += 1
    return outg[:k].copy(), tries
