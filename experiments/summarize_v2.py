"""Compact cross-experiment summary of the v2 runs: CE-loss vs 0-1-loss (accuracy) variance,
random-label controls, and the CVU baseline.

Usage: python experiments/summarize_v2.py > results/v2/summary.md
"""
import os, sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, partial_spearman  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2")
EXPS = [("grid_full", "digits, full"), ("grid_n200", "digits, n=200"), ("gain_sweep", "digits, gain sweep"),
        ("fashion_full", "Fashion, n=10k"), ("fashion_n1000", "Fashion, n=1k")]
ROWS = [
    ("CE loss", "var_init", "Var_k of CE risk (raw)"),
    ("CE loss", "var_init_norm", "Var_k of CE risk (temp.-normalized)"),
    ("CE loss", "free_energy_1_norm", "CE free energy, λ=1 (normalized; lower=better)"),
    ("CE loss", "free_energy_opt_1", "CE free energy, λ=1, optimal temperature"),
    ("0-1 loss", "acc_var_init", "Var_k of 0-1 risk (= Var of untrained acc)"),
    ("0-1 loss", "acc_var_pointwise", "E_i Var_k of per-example error"),
    ("0-1 loss", "acc_overdispersion", "error overdispersion (Var_init / binomial floor)"),
    ("0-1 loss", "acc_free_energy_1", "0-1 free energy, λ=1 (lower=better)"),
    ("0-1 loss", "acc_best", "best-of-K untrained acc"),
    ("0-1 loss", "untrained_acc", "mean untrained acc"),
    ("0-1 loss", "gauss_logmass", "Gaussian-tail log P0(err=0) ≈ -μ²/2σ² (higher=better)"),
    ("control", "acc_var_init_randlab", "Var_k of 0-1 risk, permuted labels"),
    ("control", "var_init_norm_randlab", "Var_k of CE risk (norm.), permuted labels"),
    ("baseline", "cvu", "CVU (Gracheva 2021) = std/mean of untrained acc"),
    ("baseline", "zico", "ZiCo (simplified)"),
    ("baseline", "log_n_params", "log #params"),
]


def main():
    data = {}
    for f, _ in EXPS:
        p = os.path.join(RES, f + ".jsonl")
        if os.path.exists(p):
            d = load(p)
            P = d[1]
            P["gauss_logmass"] = -P["err_mean"] ** 2 / (2 * np.maximum(P["acc_var_init"], 1e-12))
            data[f] = d
    exps = [(f, t) for f, t in EXPS if f in data]
    out = ["| family | proxy | " + " | ".join(t for _, t in exps) + " |",
           "|---|---|" + "---|" * len(exps)]
    for fam, k, desc in ROWS:
        cells = []
        for f, _ in exps:
            _, P, T, _ = data[f]
            x, y = np.nan_to_num(P[k], nan=0.0), T["test_acc"]
            if np.ptp(x) == 0:
                cells.append("—")
                continue
            r = spearmanr(x, y)[0]
            pc = partial_spearman(x, y, P["cvu"]) if k != "cvu" else float("nan")
            cells.append(f"{r:+.2f}" + ("" if np.isnan(pc) else f" ({pc:+.2f})"))
        out.append(f"| {fam} | `{k}`: {desc} | " + " | ".join(cells) + " |")
    print("Spearman ρ with mean test accuracy; in parentheses: partial ρ controlling for CVU.\n")
    print("\n".join(out))
    print()
    for f, t in exps:
        _, P, T, _ = data[f]
        y = T["test_acc"]
        print(f"- {t}: n={len(y)}, test acc median {np.median(y):.3f}, "
              f"range {y.min():.3f}-{y.max():.3f}, seed std (median) {np.median(T['test_acc_std']):.4f}")


if __name__ == "__main__":
    main()
