"""Time-uniform RDP for contractive noisy SGD (Altschuler & Talwar 2022, Thms 3.1 / 5.1, explicit constants from
their proof, eq. 3.6 and Prop. 5.6), plugged into the L-CL-Z Thm 8 tail (theory.md s16).

Algorithm (required by the theorem):  w <- Proj_K[ phi_t(w) - eta Z ],  phi_t(w) = (1 - eta lam) w - eta g_t(w),
g_t = mean over a uniformly drawn fixed-size batch (size b) of per-example gradients, Z ~ N(0, sigma^2 I),
per-example losses convex, L-Lipschitz on K, M-smooth; K a ball of diameter D; eta (2 lam + M) <= 2 so that phi_t is
c-contractive with c = 1 - eta lam.  No momentum, no clipping.
Replace-one RDP, noise split sigma^2 = s1^2 + s2^2 (Z = Z1 + Z2):
  eps(alpha) <= min_{R <= T} [ R * S_alpha(q, s2 b / (2L))  +  alpha c^{2R} D^2 / (2 eta^2 s1^2 sum_{k<R} c^{2k}) ]
(R = T, tau = 0: no PABI term, s2 = sigma.)  S_alpha = sampled-Gaussian RDP (mixture-first direction, an upper bound
on the N(0) || mixture direction AT need), integer alpha, noise multiplier relative to the replace-one sensitivity 2L/b.
"""
import math, functools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_bounds import eps_add, lz8_kappa, kl_inv  # noqa: E402

BETA, D2 = 0.024, 0.001


@functools.lru_cache(None)
def S(q, mult, a):
    return eps_add(q, mult, a)


def pabi_eps(a, q, sigma, b, T, eta, lam, D, L=1.0, splits=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95), detail=False):
    c = 1 - eta * lam
    best = (T * S(q, round(sigma * b / (2 * L), 9), a), dict(R=T, f=1.0))
    Rs = sorted(set(int(x) for x in np.unique(np.geomspace(1, max(T - 1, 1), 60).astype(int))))
    for f in splits:                      # f = s2^2 / sigma^2
        s1, s2 = sigma * math.sqrt(1 - f), sigma * math.sqrt(f)
        m2 = round(s2 * b / (2 * L), 9)
        for R in Rs:
            geo = R if c == 1 else (1 - c ** (2 * R)) / (1 - c * c)
            v = R * S(q, m2, a) + a * c ** (2 * R) * D * D / (2 * eta * eta * s1 * s1 * geo)
            if v < best[0]:
                best = (v, dict(R=R, f=f))
    return best if detail else best[0]


def certificate(train01, n, eps_fn, alphas=tuple(range(2, 257)) + (320, 400, 512, 768, 1024)):
    k, info = lz8_kappa(n, eps_fn, BETA, alphas)
    return kl_inv(train01, (k + math.log(2 * math.sqrt(n) / D2)) / n), k, info


if __name__ == "__main__":
    n, b = 55000, 4096
    q = b / n
    print("kappa (nats) under Thm 8: composition vs PABI, nu = noise std on mean grad (units of L)")
    for nu in (0.01, 0.005):
        for eta in (1.0, 4.0):
            for lam in (1e-3, 3e-3, 1e-2):
                rho = math.sqrt(2 * math.log(10) / lam) + 3 * math.sqrt(8340 * eta * nu * nu / (2 * lam))
                D = 2 * rho
                for T in (134, 1000, 10000):
                    comp = lambda a: T * S(q, round(nu * b / 2, 9), a)
                    pab = lambda a: pabi_eps(a, q, nu, b, T, eta, lam, D)
                    kc = lz8_kappa(n, comp, BETA, list(range(2, 129)) + [192, 256, 384, 512])[0]
                    kp, info = lz8_kappa(n, pab, BETA, list(range(2, 129)) + [192, 256, 384, 512])
                    det = pabi_eps(info["alpha"], q, nu, b, T, eta, lam, D, detail=True)[1]
                    print(f"nu={nu} eta={eta} lam={lam:g} D={D:.0f} T={T:5d}: kappa comp={kc:9.1f}  pabi={kp:8.1f}"
                          f"  (alpha={info['alpha']}, R={det['R']}, s2^2/s^2={det['f']})", flush=True)
