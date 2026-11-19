"""Training-free proxies computed at initialization.

Notation: theta_k ~ P0 (the init distribution = PAC-Bayes prior), k = 1..K;
l_ik = loss of example i under theta_k on a fixed probe batch of size B.

Loss-variance family (the object of study)
------------------------------------------
  mean_loss      E_k[ mean_i l_ik ]
  var_init       Var_k[ mean_i l_ik ]              between-init variance of the empirical risk
  var_data       E_k[ Var_i l_ik ]                 within-init (across-example) variance
  var_total      Var_{i,k}[ l_ik ]  = var_data + Var_k-of-example-means ... (law of total variance)
  var_pointwise  E_i[ Var_k l_ik ]                 per-example variance over inits, averaged
  cv_init        sqrt(var_init) / mean_loss
  free_energy_l  -(1/l) log E_k exp(-l * B * mean_i l_ik)  (Gibbs/PAC-Bayes log-partition, MC estimate)
  cumulant_l     mean_loss - l*B/2 * var_init      (2nd-order cumulant expansion of the above)

Baselines from the literature
-----------------------------
  n_params, cvu (Gracheva 2021: CV of untrained accuracy over inits),
  naswot (Mellor et al. 2021: logdet of activation-pattern kernel),
  grad_norm (Abdelfattah et al. 2021), zico (Li et al. 2023, simplified).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .mlp import Arch, init_params, forward, per_example_ce, loss_and_grads


def loss_matrix(arch: Arch, X, y, n_classes, K=32, seed0=10_000, normalize=False, tau=1.0):
    """Return L (K x B) per-example losses and A (K,) accuracies for K inits.

    normalize=True: per init, centre logits per example and rescale them so their overall
    std equals tau. This removes the (trivial) logit-scale effect and keeps only the
    'shape' of the random function -- which examples/classes it ranks high.
    """
    L = np.empty((K, len(y)))
    A = np.empty(K)
    with np.errstate(all="ignore"):
        for k in range(K):
            params = init_params(arch, X.shape[1], n_classes, np.random.default_rng(seed0 + k))
            logits = forward(arch, params, X)
            if normalize:
                z = logits - logits.mean(1, keepdims=True)
                s = z.std()
                logits = tau * z / s if np.isfinite(s) and s > 0 else np.zeros_like(z)
            L[k] = per_example_ce(logits, y)
            A[k] = (logits.argmax(1) == y).mean()
    return L, A


def loss_variance_proxies(L, lambdas=(0.01, 0.1, 1.0)):
    L = np.where(np.isfinite(L), L, 1e6)
    K, B = L.shape
    risk = L.mean(1)                     # empirical risk per init
    out = {
        "mean_loss": risk.mean(),
        "var_init": risk.var(ddof=1),
        "var_data": L.var(1, ddof=1).mean(),
        "var_total": L.var(ddof=1),
        "var_pointwise": L.var(0, ddof=1).mean(),
        "min_loss": risk.min(),
    }
    out["cv_init"] = np.sqrt(out["var_init"]) / out["mean_loss"]
    out["log_var_init"] = np.log(out["var_init"] + 1e-30)
    out["log_var_data"] = np.log(out["var_data"] + 1e-30)
    out["log_var_pointwise"] = np.log(out["var_pointwise"] + 1e-30)
    for lam in lambdas:
        a = -lam * B * risk
        out[f"free_energy_{lam:g}"] = -(logsumexp(a) - np.log(K)) / (lam * B)
        out[f"cumulant_{lam:g}"] = out["mean_loss"] - lam * B / 2 * out["var_init"]
    return out


def cvu(A):
    """Gracheva (2021): coefficient of variation of untrained accuracy (lower = better)."""
    return A.std(ddof=1) / max(A.mean(), 1e-12)


def naswot(arch: Arch, X, n_classes, seed=0):
    """Mellor et al. (2021) logdet of Hamming-similarity kernel of pre-activation sign patterns."""
    params = init_params(arch, X.shape[1], n_classes, np.random.default_rng(seed))
    with np.errstate(all="ignore"):
        _, _, pre = forward(arch, params, X, keep=True)
    if not pre:
        return 0.0
    C = np.concatenate([(z > 0).astype(float) for z in pre], 1)
    Kh = C @ C.T + (1 - C) @ (1 - C).T
    sign, ld = np.linalg.slogdet(Kh)
    return float(ld) if sign > 0 else -1e3


def grad_proxies(arch: Arch, X, y, n_classes, n_batches=4, seed=0):
    """grad_norm (single batch) and a simplified ZiCo over n_batches minibatches."""
    params = init_params(arch, X.shape[1], n_classes, np.random.default_rng(seed))
    chunks = np.array_split(np.arange(len(y)), n_batches)
    G = []
    with np.errstate(all="ignore"):
        for c in chunks:
            _, g = loss_and_grads(arch, params, X[c], y[c])
            G.append(g)
    gn = np.sqrt(sum((w ** 2).sum() for layer in G[0] for w in layer))
    zico = 0.0
    for li in range(len(params)):
        Wg = np.stack([G[b][li][0] for b in range(n_batches)])  # (nb, fi, fo)
        mu, sd = np.abs(Wg.mean(0)), Wg.std(0)
        ratio = (mu / (sd + 1e-12))
        s = ratio[sd > 0].sum()
        if s > 0 and np.isfinite(s):
            zico += np.log(s)
    return {"grad_norm": float(gn) if np.isfinite(gn) else 0.0, "zico": float(zico)}


def all_proxies(arch: Arch, X, y, n_classes, K=32):
    L, A = loss_matrix(arch, X, y, n_classes, K=K)
    out = loss_variance_proxies(L)
    Ln, _ = loss_matrix(arch, X, y, n_classes, K=K, normalize=True)
    for k, v in loss_variance_proxies(Ln).items():
        out[f"{k}_norm"] = v
    out["cvu"] = cvu(A)
    out["untrained_acc"] = A.mean()
    out["n_params"] = arch.n_params(X.shape[1], n_classes)
    out["log_n_params"] = np.log(out["n_params"])
    out["naswot"] = naswot(arch, X[:64], n_classes)
    out.update(grad_proxies(arch, X, y, n_classes))
    return {k: float(v) for k, v in out.items()}
