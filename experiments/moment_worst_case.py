"""Worst case of Catoni's optimized bound over all prior-risk distributions on [0,1] with given mean and variance.
Catoni (Picard-Weibel convention): min_rho B = -lam*log E_P0[exp(-R/lam)] - lam*log(delta) + 1/(8 n lam).
Extreme points of {p : E1=1, ER=mu, ER^2=s2} have <= 3 atoms, so enumerate all 3-atom distributions on a grid
and compute log E exp(-R/lam) exactly in log space."""
import itertools, numpy as np
from scipy.special import logsumexp

G = np.round(np.linspace(0, 1, 201), 6)
T = np.array(list(itertools.combinations(range(len(G)), 3)))
X = G[T]                                                    # (N,3) atoms

def feasible(mu, var):
    m2 = var + mu**2
    A = np.stack([np.ones_like(X), X, X**2], 1)             # (N,3,3)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    W = np.full_like(X, -1.0)
    W[ok] = np.linalg.solve(A[ok], np.broadcast_to([1, mu, m2], (ok.sum(), 3))[..., None])[..., 0]
    keep = (W >= -1e-12).all(1)
    # two-atom distributions (a weight ~0) are included as limits of these
    return X[keep], np.clip(W[keep], 1e-300, None)

def worst(mu, var, lams, n, delta):
    x, w = feasible(mu, var)
    best = []
    for lam in lams:
        logE = logsumexp(-x / lam, b=w, axis=1)             # log E exp(-R/lam) for each feasible dist
        i = np.argmin(logE)                                 # worst case = smallest MGF
        best.append((-lam * logE[i] - lam * np.log(delta) + 1 / (8 * n * lam), lam, x[i], w[i]))
    return min(best, key=lambda t: t[0])

mu, delta = 0.9, 0.05
lams = np.logspace(-3.5, 0, 80)
print("worst-case Catoni bound given ONLY prior mean (0.9, pinned by symmetry) and prior sd; trivial value 0.9")
for n in [256, 1347, 60000]:
    out = []
    for sd in [0.01, 0.03, 0.04, 0.05, 0.10, 0.20]:
        b, lam, x, w = worst(mu, sd**2, lams, n, delta)
        out.append(f"sd {sd:.2f}: {b:.3f}")
    print(f"n={n:6d} | " + " | ".join(out))
b, lam, x, w = worst(mu, 0.04**2, lams, 1347, delta)
print("\nextremal distribution at sd=0.04, n=1347:", dict(zip(np.round(x, 3), np.round(w, 3))), "lam", round(lam, 4))
