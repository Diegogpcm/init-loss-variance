"""Exact (floating-point-free) lattice HMC for the prequential AIS certificate (theory.md s17.5).

Why: the certificate needs each MCMC kernel to be EXACTLY reversible w.r.t. its stage target. In floating point the
state space is a finite grid, and a leapfrog step that is not an exact bijection on that grid makes the path ratio
infinite rather than slightly wrong, so the error cannot be "bounded"; it has to be removed. (The same issue for
differential privacy: Mironov 2012, fixed by the discrete Gaussian of Canonne, Kamath & Steinke 2020.)

Construction. State (i, j) in Z^D x Z^D; weights w = dx * i (dx = 2^-24, exact in float64), momenta p_g = dp_g * j.
  prior      P(i) = prod_d DG(i_d; s2x_g)       discrete Gaussian, s2x_g = v_g / dx^2 (exact rational), data-free
  momentum   mu(j) = prod_d DG(j_d; Np)        Np = 2^24 (integer), sampled exactly, vectorised (int64)
  energy     E_k(i) = sum_d i_d^2 / (2 s2x_g) + lam * sum_{blocks b <= k} bhat_b(w)     (exact rational)
             bhat_b(w) = float64 sum of the cross-entropies of block b, computed by one fixed call per block; any
             deterministic function of i is allowed, it just defines the target.
  kinetic    K(j) = sum_d j_d^2 / (2 Np)       (exact rational)
  proposal   palindromic leapfrog of integer shears, then a momentum flip:
               j <- j - rint(h ghat(w) / dp)          (ghat: float64 gradient, any deterministic function of i)
               i <- i + rint(c_g * j),  c_g = h v_g dp_g / dx  (float; rint and float product are odd-symmetric)
             each shear is exactly invertible, so the flipped composition is an exact involution on Z^2D.
  accept     with probability min(1, exp(E0 + K0 - E1 - K1)), by an exact Bernoulli(exp(-gamma)) sampler on the
             exact rational gamma.
Then each kernel is exactly reversible w.r.t. pi_k(i) ~ P(i) exp(-lam sum_{b<=k} bhat_b) and the certificate's
C = lam [ sum_k bhat_k(w_{k-1}) - sum_b bhat_b(w_K) ] is an exact rational. Remaining assumptions: ideal random bits,
and determinism of ghat / bhat (audited: recomputed bit-for-bit at the start of every stage; explicit round trips).
"""
import math, random
from fractions import Fraction
import numpy as np

DX_EXP = 24
DX = 2.0 ** -DX_EXP
NP = 2 ** 24                                              # momentum lattice variance (sigma = 4096)


# ---------- exact samplers (Canonne, Kamath & Steinke 2020, Algorithms 1-3) ----------
def bern_exp(gamma, rng):
    """exact Bernoulli(exp(-gamma)), gamma a non-negative Fraction; rng: random.Random"""
    gamma = Fraction(gamma)
    while gamma > 1:
        if not bern_exp(Fraction(1), rng):
            return False
        gamma -= 1
    k = 1
    while True:                                           # A_k ~ Bernoulli(gamma / k) until the first 0
        if rng.randrange(gamma.denominator * k) >= gamma.numerator:
            break
        k += 1
    return k % 2 == 1


def dg_sample_exact(sigma2, rng):
    """one exact discrete Gaussian sample, P(y) ~ exp(-y^2 / (2 sigma2)), sigma2 a positive Fraction (big ints ok)"""
    sigma2 = Fraction(sigma2)
    t = math.isqrt(sigma2.numerator // sigma2.denominator) + 1
    while True:
        while True:                                       # discrete Laplace(t)
            u = rng.randrange(t)
            if not bern_exp(Fraction(u, t), rng):
                continue
            v = 0
            while bern_exp(Fraction(1), rng):
                v += 1
            b = rng.randrange(2)
            x = u + t * v
            if b == 1 and x == 0:
                continue
            y = -x if b else x
            break
        g = (abs(y) - sigma2 / t) ** 2 / (2 * sigma2)
        if bern_exp(g, rng):
            return y


def _bern_exp_frac_vec(num, den, rng):
    """vectorised exact Bernoulli(exp(-num/den)) for int64 arrays num >= 0, den > 0 (no overflow by construction)"""
    out = np.ones(num.shape, bool)
    q, r = num // den, num % den
    # floor part: q independent Bernoulli(exp(-1)) must all succeed
    need = q.copy()
    while True:
        m = out & (need > 0)
        if not m.any():
            break
        idx = np.nonzero(m)[0]
        out[idx] &= _bern_exp1_vec(len(idx), rng)
        need[idx] -= 1
    # fractional part r/den in [0, 1): K = first k with A_k = 0, A_k ~ Bernoulli(r / (den k)); success iff K odd
    idx = np.nonzero(out)[0]
    k = np.ones(len(idx), np.int64)
    alive = np.ones(len(idx), bool)
    rr, dd = r[idx], den[idx]
    while alive.any():
        a = np.nonzero(alive)[0]
        draw = rng.integers(0, dd[a] * k[a])
        cont = draw < rr[a]
        k[a[cont]] += 1
        alive[a[~cont]] = False
    out[idx] = (k % 2) == 1
    return out


def _bern_exp1_vec(n, rng):
    k = np.ones(n, np.int64)
    alive = np.ones(n, bool)
    while alive.any():
        a = np.nonzero(alive)[0]
        cont = rng.integers(0, k[a]) < 1
        k[a[cont]] += 1
        alive[a[~cont]] = False
    return (k % 2) == 1


def dg_sample_vec(N, size, rng):
    """size exact samples of the discrete Gaussian with integer variance parameter N (<= 2^24), int64 arithmetic"""
    assert N <= 2 ** 24
    t = math.isqrt(N) + 1
    out = np.empty(size, np.int64)
    todo = np.arange(size)
    while len(todo):
        m = len(todo)
        y = np.empty(m, np.int64)
        pend = np.arange(m)
        while len(pend):                                  # discrete Laplace(t), vectorised rejection
            u = rng.integers(0, t, len(pend))
            okd = _bern_exp_frac_vec(u, np.full(len(pend), t, np.int64), rng)
            v = np.zeros(len(pend), np.int64)
            alive = okd.copy()
            while alive.any():
                a = np.nonzero(alive)[0]
                s = _bern_exp1_vec(len(a), rng)
                v[a[s]] += 1
                alive[a[~s]] = False
            b = rng.integers(0, 2, len(pend))
            x = u + t * v
            good = okd & ~((b == 1) & (x == 0))
            y[pend[good]] = np.where(b[good] == 1, -x[good], x[good])
            pend = pend[~good]
        ay = np.abs(y)
        assert ay.max() * t < 3_000_000_000, "discrete Laplace tail beyond the int64-safe range"
        num = (ay * t - N) ** 2
        den = np.full(m, 2 * N * t * t, np.int64)
        acc = _bern_exp_frac_vec(num, den, rng)
        out[todo[acc]] = y[acc]
        todo = todo[~acc]
    return out


# ---------- exact lattice AIS ----------
class ExactAIS:
    """R independent exact lattice-HMC AIS chains (no resampling). Model given by logits(w (B, D), X) and prior
    variances pv (one float per coordinate, constant within groups given by `groups` = list of (start, end, v as
    Fraction))."""

    def __init__(self, torch, X, y, logits, groups, lam, bounds, S, L, c, n0, seed, chunk=10000, audit=True):
        self.t, self.X, self.y, self.logits = torch, X, y, logits
        self.groups, self.lam, self.bounds = groups, Fraction(lam).limit_denominator(10 ** 6), bounds
        self.S, self.L, self.c, self.n0, self.chunk, self.audit = S, L, c, n0, chunk, audit
        self.prng = random.Random(seed)                   # big-int exact draws (prior, MH)
        self.nrng = np.random.default_rng(seed)           # vectorised exact draws (momenta)
        D = groups[-1][1]
        self.D = D
        v = np.empty(D)
        for a, b, vg in groups:
            v[a:b] = float(vg)
        self.v = v
        self.dp = 1.0 / np.sqrt(v * NP)                   # momentum lattice spacing: Var(p) = dp^2 Np ~= 1/v (mass 1/v)
        self.s2x = [Fraction(vg) / Fraction(DX) ** 2 for _, _, vg in groups]
        self.stats = dict(audits=0, audit_fail=0, roundtrips=0, roundtrip_fail=0)

    # ---- exact energies ----
    def prior_energy(self, i):                            # sum_d i_d^2 / (2 s2x_g), exact
        e = Fraction(0)
        for (a, b, _), s2 in zip(self.groups, self.s2x):
            e += Fraction(int(np.sum(i[a:b].astype(object) ** 2)), 1) / (2 * s2)
        return e

    @staticmethod
    def kinetic(j):
        return Fraction(int(np.sum(j.astype(object) ** 2)), 2 * NP)

    def block_losses(self, i, kmax):                      # [bhat_1, ..., bhat_kmax] as python floats (exact values)
        t = self.t
        w = t.from_numpy(i.astype(np.float64) * DX)[None]
        out = []
        with t.no_grad():
            for k in range(1, kmax + 1):
                lo, hi = self.bounds[k - 1], self.bounds[k]
                lg = self.logits(w, self.X[lo:hi])[0]
                ce = t.logsumexp(lg, 1) - lg.gather(1, self.y[lo:hi][:, None])[:, 0]
                out.append(math.fsum(ce.tolist()))        # exactly rounded sum: deterministic given ce
        return out

    def energy(self, i, k):
        bl = self.block_losses(i, k)
        return self.prior_energy(i) + self.lam * sum(Fraction(b) for b in bl), bl

    def grad(self, i, k):                                 # float64 gradient of the energy (any deterministic fn ok)
        t = self.t
        w = t.from_numpy(i.astype(np.float64) * DX)[None].requires_grad_()
        hi = self.bounds[k]
        u = 0
        for s in range(0, hi, self.chunk):
            e = min(hi, s + self.chunk)
            lg = self.logits(w, self.X[s:e])[0]
            u = u + (t.logsumexp(lg, 1) - lg.gather(1, self.y[s:e][:, None])[:, 0]).sum()
        g, = t.autograd.grad(float(self.lam) * u, w)
        return g[0].numpy() + i.astype(np.float64) * DX / self.v

    # ---- integrator: palindromic integer shears + flip (exact involution) ----
    def leapfrog(self, i, j, k, h):
        cg = h * self.v * self.dp / DX
        i, j = i.copy(), j.copy()
        j -= np.rint(0.5 * h * self.grad(i, k) / self.dp).astype(np.int64)
        for l in range(self.L):
            i += np.rint(cg * j).astype(np.int64)
            hh = h if l < self.L - 1 else 0.5 * h
            j -= np.rint(hh * self.grad(i, k) / self.dp).astype(np.int64)
        return i, -j

    def run(self, on_stage=None, roundtrip_every=0):
        K = len(self.bounds) - 1
        i = np.array([dg_sample_exact(self.s2x[gi], self.prng) for gi, (a, b, _) in enumerate(self.groups)
                      for _ in range(b - a)], dtype=np.int64)
        online = []
        acc_log = []
        E, bl = None, None
        for k in range(1, K + 1):
            bk = self.block_losses(i, k)                  # online charge: block k at the pre-move state
            if bl is not None and self.audit:             # determinism audit: recompute old blocks bit-for-bit
                self.stats["audits"] += 1
                if bk[:k - 1] != bl[:k - 1]:
                    self.stats["audit_fail"] += 1
            online.append(bk[k - 1])
            E = self.prior_energy(i) + self.lam * sum(Fraction(b) for b in bk)
            h = self.c / math.sqrt(1 + float(self.lam) * max(self.bounds[k], self.n0))
            acc = 0
            for s in range(self.S):
                j = dg_sample_vec(NP, self.D, self.nrng)
                i1, j1 = self.leapfrog(i, j, k, h)
                if roundtrip_every and (s % roundtrip_every == 0):
                    self.stats["roundtrips"] += 1
                    i2, j2 = self.leapfrog(i1, j1, k, h)
                    if not (np.array_equal(i2, i) and np.array_equal(j2, j)):
                        self.stats["roundtrip_fail"] += 1
                E1, bl1 = self.energy(i1, k)
                d = (E + self.kinetic(j)) - (E1 + self.kinetic(j1))      # exact rational log acceptance ratio
                if d >= 0 or bern_exp(-d, self.prng):
                    i, E, bk = i1, E1, bl1
                    acc += 1
            bl = bk
            acc_log.append(acc / self.S)
            if on_stage is not None:
                on_stage(k, dict(k=k, n_k=self.bounds[k], eps=h, acc=acc / self.S), i)
        final = self.block_losses(i, K)
        C = self.lam * (sum(Fraction(b) for b in online) - sum(Fraction(b) for b in final))
        return dict(i=i, w=i.astype(np.float64) * DX, C=C, online=online, final=final, acc=acc_log, stats=self.stats)
