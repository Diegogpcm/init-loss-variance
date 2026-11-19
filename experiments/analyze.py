"""Correlate init-time proxies with trained test accuracy.

Reports, for each proxy:
  rho_all     Spearman vs test acc, all architectures
  rho_ok      Spearman on 'trainable' archs (test acc > 0.5) - removes the easy broken-vs-working split
  pcor_mean   partial Spearman controlling for mean init loss (is variance informative beyond scale?)
  ci          95% bootstrap CI for rho_all
  pcor_cvu    partial Spearman controlling for CVU (anything beyond Gracheva 2021?)
  rho_strat   mean within-activation Spearman (controls for activation family)
"""
import argparse, json, sys

import numpy as np
from scipy.stats import spearmanr, kendalltau, rankdata

PROXIES = ["var_init", "log_var_init", "var_data", "log_var_data", "var_pointwise", "var_total",
           "cv_init", "mean_loss", "min_loss", "free_energy_0.01", "free_energy_0.1", "free_energy_1",
           "cumulant_0.01",
           "mean_loss_norm", "var_init_norm", "var_data_norm", "var_pointwise_norm", "cv_init_norm",
           "free_energy_0.1_norm", "free_energy_1_norm",
           # 0-1 loss (accuracy) family
           "err_mean", "acc_var_init", "acc_var_pointwise", "acc_overdispersion", "acc_best",
           "acc_free_energy_0.1", "acc_free_energy_1", "acc_free_energy_10",
           # controls
           "acc_var_init_randlab", "acc_overdispersion_randlab", "var_init_norm_randlab",
           "free_energy_opt_0.1", "free_energy_opt_1",
           # baselines
           "cvu", "untrained_acc", "log_n_params", "naswot", "grad_norm", "zico"]


def load(path):
    rows = [json.loads(l) for l in open(path)]
    P = {k: np.array([r["proxies"].get(k, np.nan) for r in rows]) for k in PROXIES}
    T = {k: np.array([r["trained"][k] for r in rows]) for k in rows[0]["trained"]}
    T["gen_gap"] = T["train_acc"] - T["test_acc"]
    acts = np.array([r["arch"]["act"] for r in rows])
    return rows, P, T, acts


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(x, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        vals.append(spearmanr(x[i], y[i])[0])
    return np.nanpercentile(vals, [2.5, 97.5])


def table(path, target="test_acc"):
    rows, P, T, acts = load(path)
    y = T[target]
    ok = T["test_acc"] > 0.5
    out = []
    for k in PROXIES:
        if np.isnan(P[k]).all():
            continue
        x = np.nan_to_num(P[k], nan=0.0, posinf=1e30, neginf=-1e30)
        r_all = spearmanr(x, y)[0]
        tau = kendalltau(x, y)[0]
        r_ok = spearmanr(x[ok], y[ok])[0]
        pc = partial_spearman(x, y, P["mean_loss"]) if k != "mean_loss" else float("nan")
        pcc = partial_spearman(x, y, P["cvu"]) if k != "cvu" else float("nan")
        lo, hi = boot_ci(x, y)
        strat = [spearmanr(x[acts == a], y[acts == a])[0] for a in np.unique(acts) if (acts == a).sum() > 5]
        out.append(dict(proxy=k, rho_all=r_all, tau_all=tau, ci_lo=lo, ci_hi=hi, rho_ok=r_ok,
                        pcor_mean=pc, pcor_cvu=pcc, rho_strat=float(np.nanmean(strat))))
    return out, dict(n=len(y), n_ok=int(ok.sum()), acc_mean=float(y.mean()), acc_min=float(y.min()),
                     acc_max=float(y.max()))


def fmt(out, meta, title):
    s = [f"### {title}", "", f"n = {meta['n']} architectures ({meta['n_ok']} with test acc > 0.5); "
         f"test acc range {meta['acc_min']:.3f}-{meta['acc_max']:.3f}", "",
         "| proxy | Spearman (all) | 95% CI | Kendall | Spearman (acc>0.5) | partial: mean loss | partial: CVU | within-act |",
         "|---|---|---|---|---|---|---|---|"]
    for o in out:
        s.append(f"| {o['proxy']} | {o['rho_all']:+.3f} | [{o['ci_lo']:+.2f}, {o['ci_hi']:+.2f}] | "
                 f"{o['tau_all']:+.3f} | {o['rho_ok']:+.3f} | {o['pcor_mean']:+.3f} | {o['pcor_cvu']:+.3f} | {o['rho_strat']:+.3f} |")
    return "\n".join(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--target", default="test_acc")
    a = ap.parse_args()
    for f in a.files:
        out, meta = table(f, a.target)
        print(fmt(out, meta, f"{f} -> {a.target}"))
        print()
