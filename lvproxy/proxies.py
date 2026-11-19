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

0-1 loss (accuracy) family  -- e_ik = 1[argmax f(x_i; theta_k) != y_i]
-------------------------------------------------------------------------
  err_mean            E_k[ mean_i e_ik ]               = 1 - untrained_acc
  acc_var_init        Var_k[ mean_i e_ik ]             between-init variance of the 0-1 risk (= Var of untrained acc)
  acc_std_init        sqrt(acc_var_init)
  acc_var_pointwise   E_i[ p_i (1 - p_i) ], p_i = P_k(e_ik = 1)   per-example variance over inits
  acc_overdispersion  acc_var_init / (acc_var_pointwise / B)  = 1 + (B-1) * mean inter-example correlation
                      of errors under the prior; 1 means examples are flipped independently
  acc_free_energy_l   -(1/l) log E_k exp(-l * B * err_k)  (0-1 loss is bounded, so this is the quantity
                      the Catoni bound actually uses)
  acc_best            max_k acc_k  (the l -> inf limit)
  Note: cvu = acc_std_init / untrained_acc, so with untrained_acc ~ 1/C, CVU is ~ the 0-1 between-init std.

Controls
--------
  *_randlab           same quantity with the probe labels randomly permuted (label-independent part)
  free_energy_opt_l   min over temperature T of F_T(l), with logits normalized to std T
  T_opt_l             the minimizing temperature

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


def init_logits(arch: Arch, X, n_classes, K=32, seed0=10_000):
    """Logits (K, B, C) for K inits drawn from the prior P0."""
    Z = np.empty((K, len(X), n_classes))
    with np.errstate(all="ignore"):
        for k in range(K):
            params = init_params(arch, X.shape[1], n_classes, np.random.default_rng(seed0 + k))
            Z[k] = forward(arch, params, X)
    return Z


def normalize_logits(Z, tau=1.0):
    """Per init: centre logits per example, rescale overall std to tau (same as loss_matrix)."""
    out = np.zeros_like(Z)
    with np.errstate(all="ignore"):
        for k in range(len(Z)):
            z = Z[k] - Z[k].mean(1, keepdims=True)
            s = z.std()
            if np.isfinite(s) and s > 0:
                out[k] = tau * z / s
    return out


def ce_matrix(Z, y):
    with np.errstate(all="ignore"):
        return np.stack([per_example_ce(z, y) for z in Z])


def err_matrix(Z, y):
    E = (Z.argmax(2) != y[None, :]).astype(float)
    E[~np.isfinite(Z).all(2)] = 1.0 - 1.0 / Z.shape[2]   # non-finite logits: chance-level error
    return E


def zero_one_proxies(E, lambdas=(0.1, 1.0, 10.0)):
    K, B = E.shape
    risk = E.mean(1)
    p = E.mean(0)
    out = {
        "err_mean": risk.mean(),
        "acc_var_init": risk.var(ddof=1),
        "acc_var_pointwise": (p * (1 - p)).mean() * K / (K - 1),
        "acc_best": 1 - risk.min(),
    }
    out["acc_std_init"] = np.sqrt(out["acc_var_init"])
    out["log_acc_var_init"] = np.log(out["acc_var_init"] + 1e-12)
    floor = out["acc_var_pointwise"] / B
    out["acc_overdispersion"] = out["acc_var_init"] / floor if floor > 0 else 1.0
    for lam in lambdas:
        a = -lam * B * risk
        out[f"acc_free_energy_{lam:g}"] = -(logsumexp(a) - np.log(K)) / (lam * B)
    return out


def temperature_opt_free_energy(Zn, y, lambdas=(0.1, 1.0), temps=(0.03, 0.06, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16)):
    """min_T F_T(l), where F_T uses CE of unit-std-normalized logits scaled to std T."""
    K, B = Zn.shape[:2]
    risks = np.stack([np.nan_to_num(ce_matrix(T * Zn, y), nan=1e6, posinf=1e6).mean(1) for T in temps])
    out = {}
    for lam in lambdas:
        F = -(logsumexp(-lam * B * risks, axis=1) - np.log(K)) / (lam * B)
        j = int(np.argmin(F))
        out[f"free_energy_opt_{lam:g}"] = F[j]
        out[f"T_opt_{lam:g}"] = temps[j]
    return out


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


def all_proxies(arch: Arch, X, y, n_classes, K=32, perm_seed=123):
    Z = init_logits(arch, X, n_classes, K=K)
    Zn = normalize_logits(Z)
    y_rand = np.random.default_rng(perm_seed).permutation(y)

    out = loss_variance_proxies(ce_matrix(Z, y))
    for k, v in loss_variance_proxies(ce_matrix(Zn, y)).items():
        out[f"{k}_norm"] = v
    E = err_matrix(Z, y)
    out.update(zero_one_proxies(E))
    out.update(temperature_opt_free_energy(Zn, y))

    # random-label controls (label-independent part of each quantity)
    for k, v in zero_one_proxies(err_matrix(Z, y_rand)).items():
        out[f"{k}_randlab"] = v
    r = loss_variance_proxies(ce_matrix(Zn, y_rand))
    out["var_init_norm_randlab"] = r["var_init"]
    out["free_energy_1_norm_randlab"] = r["free_energy_1"]

    A = 1 - E.mean(1)
    out["cvu"] = cvu(A)
    out["untrained_acc"] = A.mean()
    out["n_params"] = arch.n_params(X.shape[1], n_classes)
    out["log_n_params"] = np.log(out["n_params"])
    out["naswot"] = naswot(arch, X[:64], n_classes)
    out.update(grad_proxies(arch, X, y, n_classes))
    return {k: float(v) for k, v in out.items()}
