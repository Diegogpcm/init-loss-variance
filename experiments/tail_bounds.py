"""High-probability bounds on the information density of Poisson-subsampled DP-SGD.

ı(S;W) = ln dQ_S/dP*(W), with P* the output marginal (the oracle prior). The single-network certificate is
    kl(train || risk) <= (kappa + ln(2 sqrt(n)/d2)) / n,   where P[ı > kappa] <= beta.
Variants of kappa (all depend only on hyperparameters n, q, sigma, T and the confidence split, so optimizing over
orders alpha and splits beta1/beta2 is free -- no union bound):

  markov : kappa = (I + 1/e)/beta,  I <= n T KL_step(q, sigma)                     (our theory.md s14)
  lz7    : Lampert-Cairney-Leeming-Zakerinia (arXiv:2605.26222v2) Thm 7:  kappa = n (eps + ln(1/beta)/(alpha-1))
  lz8    : L-CL-Z Thm 8:  kappa = v* + 2 sqrt(v* L2) + L2/(alpha-1),
               v* = n/(2(alpha-1)^2) * psi((e^{(alpha-1) eps} - 1)/beta1),  psi(u) = asinh(sqrt u)^2,  L2 = ln(1/beta2)
           eps = replace-one RDP of the whole run at order alpha.
Replace-one RDP of the run:
  conv   : (rigorous, as in L-CL-Z App. C) weak triangle inequality from add/remove:
               eps_rep(a) = (2a-1)/(2(a-1)) T eps_add(2a) + T eps_add(2a-1),
           eps_add(integer a) = ln A_a/(a-1), A_a = sum_k C(a,k) (1-q)^{a-k} q^k exp((k^2-k)/(2 sigma^2))  (Mironov et al. 2019)
  direct : per-step D_a((1-q)N(0,s^2)+qN(h,s^2) || (1-q)N(0,s^2)+qN(h',s^2)) at the antipodal pair h=-h'=1
           (worst case over |h|,|h'|<=1 checked numerically in check_direct_worst_case), composed over T steps.
"""
import math
import numpy as np


def logsumexp(v):
    v = np.asarray(v, dtype=float)
    m = v.max()
    return m + math.log(np.exp(v - m).sum())


_LF = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 200001)))])  # ln k!


def log_A_int(q, sigma, a):
    k = np.arange(a + 1)
    logc = _LF[a] - _LF[k] - _LF[a - k]
    terms = logc + (a - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
    return float(logsumexp(terms))


def eps_add(q, sigma, a):
    """add/remove RDP of ONE step at integer order a >= 2 (Mironov, Talwar, Zhang 2019)."""
    return log_A_int(q, sigma, int(a)) / (a - 1)


def eps_rep_conv(q, sigma, T, a):
    """replace-one RDP of T steps at order a (2a must be an integer), via the weak triangle inequality."""
    return (2 * a - 1) / (2 * (a - 1)) * T * eps_add(q, sigma, 2 * a) + T * eps_add(q, sigma, 2 * a - 1)


def _grid(sigma, a):
    half = (8 + math.sqrt(2 * math.log(1e300) / max(a, 1))) * sigma  # wide enough for the tilted integrand
    return np.linspace(-half - 1, half + 1, 200001)


def eps_rep_direct_step(q, sigma, a, h=1.0, hp=-1.0):
    """per-step replace-one Renyi divergence at order a for 1-D centres h, hp (numerical quadrature, log-space)."""
    x = _grid(sigma, a)
    dx = x[1] - x[0]
    lg = lambda m: -(x - m) ** 2 / (2 * sigma ** 2) - 0.5 * math.log(2 * math.pi * sigma ** 2)
    lp = np.logaddexp(math.log1p(-q) + lg(0.0), math.log(q) + lg(h))
    lq = np.logaddexp(math.log1p(-q) + lg(0.0), math.log(q) + lg(hp))
    return float(logsumexp(a * lp + (1 - a) * lq) + math.log(dx)) / (a - 1)


def check_direct_worst_case(q, sigma, a, n_ang=13):
    """2-D check: D_a for |h|=r1, |h'|=r2 at angle phi; returns (max value, argmax config, antipodal value)."""
    s = 6 * sigma
    g = np.linspace(-s - 1.5, s + 1.5, 1201)
    X, Y = np.meshgrid(g, g, indexing="ij")
    dA = (g[1] - g[0]) ** 2
    lg = lambda mx, my: -((X - mx) ** 2 + (Y - my) ** 2) / (2 * sigma ** 2) - math.log(2 * math.pi * sigma ** 2)
    best = (-1, None)
    for r1 in (0.5, 1.0):
        for r2 in (0.5, 1.0):
            for phi in np.linspace(0, math.pi, n_ang):
                lp = np.logaddexp(math.log1p(-q) + lg(0, 0), math.log(q) + lg(r1, 0))
                lq = np.logaddexp(math.log1p(-q) + lg(0, 0), math.log(q) + lg(r2 * math.cos(phi), r2 * math.sin(phi)))
                d = float(logsumexp(a * lp + (1 - a) * lq) + math.log(dA)) / (a - 1)
                if d > best[0]:
                    best = (d, (r1, r2, round(float(phi), 3)))
    return best


def psi(u):
    return math.asinh(math.sqrt(u)) ** 2


def lz8_kappa(n, eps_fn, beta, alphas, b1_fracs=np.linspace(0.05, 0.95, 19)):
    """min over alpha, beta1 of L-CL-Z Thm 8 kappa. eps_fn(alpha) = replace-one RDP of the run."""
    best = (math.inf, None)
    for a in alphas:
        e = eps_fn(a)
        for f in b1_fracs:
            b1, b2 = f * beta, (1 - f) * beta
            x = (a - 1) * e
            if x < 600:
                ps = psi(math.expm1(x) / b1)
            else:  # asinh(sqrt u) = ln(sqrt u + sqrt(1+u)), u = e^x/b1 huge
                ps = (math.log(2) + 0.5 * (x - math.log(b1))) ** 2
            vs = n / (2 * (a - 1) ** 2) * ps
            L2 = math.log(1 / b2)
            k = vs + 2 * math.sqrt(vs * L2) + L2 / (a - 1)
            if k < best[0]:
                best = (k, dict(alpha=a, beta1=b1, v_star=vs, eps=e))
    return best


def lz7_kappa(n, eps_fn, beta, alphas):
    return min((n * (eps_fn(a) + math.log(1 / beta) / (a - 1)), a) for a in alphas)


def kl_inv(p, c, tol=1e-12):
    if c <= 0:
        return p
    lo, hi = p, 1.0
    kl = lambda a, b: (a * math.log(a / b) if a > 0 else 0) + ((1 - a) * math.log((1 - a) / (1 - b)) if a < 1 else 0)
    while hi - lo > tol:
        m = (lo + hi) / 2
        if kl(p, m) > c:
            hi = m
        else:
            lo = m
    return hi


if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--train01", type=float, default=0.0616)
    ap.add_argument("--I", type=float, default=None, help="I_ub from the run (else recomputed from KL_step)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    n, B = a.n, a.batch
    q, sigma = B / n, a.nu * B
    T = int(round(a.epochs / q))
    beta, d2 = 0.024, 0.001
    pb = math.log(2 * math.sqrt(n) / d2)
    alphas_half = [k / 2 for k in range(3, 20001)]
    out = {}
    if a.I is None:
        sys.path.insert(0, "experiments")
        from dp_sgd_bound import kl_step
        a.I = n * T * kl_step(q, sigma)
    out["markov"] = (a.I + 1 / math.e) / beta
    out["lz7_conv"] = lz7_kappa(n, lambda al: eps_rep_conv(q, sigma, T, al), beta, alphas_half[::4])[0]
    k8, info8 = lz8_kappa(n, lambda al: eps_rep_conv(q, sigma, T, al), beta, alphas_half[::4])
    out["lz8_conv"] = k8
    print(f"n={n} q={q:.4f} sigma={sigma:.2f} T={T} I={a.I:.2f}")
    print("lz8_conv argmin:", info8)
    for k, v in out.items():
        print(f"{k:10s} kappa={v:9.1f}  kl-budget={(v + pb) / n:.5f}  cert={kl_inv(a.train01, (v + pb) / n):.4f}")
    print(f"no-Markov  kappa={a.I:9.1f}  cert={kl_inv(a.train01, (a.I + pb) / n):.4f}  (not a certificate)")
