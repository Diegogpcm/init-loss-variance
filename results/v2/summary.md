Spearman ρ with mean test accuracy; in parentheses: partial ρ controlling for CVU.

| family | proxy | digits, full | digits, n=200 | digits, gain sweep | Fashion, n=10k | Fashion, n=1k |
|---|---|---|---|---|---|---|
| CE loss | `var_init`: Var_k of CE risk (raw) | -0.03 (-0.04) | -0.05 (-0.06) | +0.13 (+0.14) | +0.01 (+0.00) | +0.10 (+0.08) |
| CE loss | `var_init_norm`: Var_k of CE risk (temp.-normalized) | +0.26 (-0.10) | +0.33 (-0.10) | +0.32 (-0.25) | +0.04 (-0.23) | +0.21 (-0.16) |
| CE loss | `free_energy_1_norm`: CE free energy, λ=1 (normalized; lower=better) | -0.20 (+0.16) | -0.27 (+0.15) | -0.31 (+0.14) | -0.05 (+0.15) | -0.21 (+0.08) |
| CE loss | `free_energy_opt_1`: CE free energy, λ=1, optimal temperature | -0.22 (+0.14) | -0.28 (+0.14) | -0.26 (+0.26) | -0.04 (+0.18) | -0.22 (+0.08) |
| 0-1 loss | `acc_var_init`: Var_k of 0-1 risk (= Var of untrained acc) | +0.32 (-0.17) | +0.41 (-0.16) | +0.50 (+0.08) | +0.22 (+0.16) | +0.36 (+0.16) |
| 0-1 loss | `acc_var_pointwise`: E_i Var_k of per-example error | -0.27 (-0.25) | -0.24 (-0.22) | +0.33 (+0.13) | +0.09 (+0.12) | +0.07 (+0.13) |
| 0-1 loss | `acc_overdispersion`: error overdispersion (Var_init / binomial floor) | +0.34 (-0.19) | +0.43 (-0.18) | +0.49 (-0.03) | +0.20 (+0.13) | +0.35 (+0.13) |
| 0-1 loss | `acc_free_energy_1`: 0-1 free energy, λ=1 (lower=better) | -0.32 (-0.05) | -0.39 (-0.06) | -0.57 (-0.33) | -0.24 (-0.15) | -0.36 (-0.16) |
| 0-1 loss | `acc_best`: best-of-K untrained acc | +0.32 (+0.05) | +0.39 (+0.06) | +0.57 (+0.34) | +0.24 (+0.15) | +0.36 (+0.16) |
| 0-1 loss | `untrained_acc`: mean untrained acc | -0.28 (-0.26) | -0.25 (-0.23) | +0.30 (+0.07) | +0.09 (+0.11) | +0.07 (+0.12) |
| 0-1 loss | `gauss_logmass`: Gaussian-tail log P0(err=0) ≈ -μ²/2σ² (higher=better) | +0.32 (-0.17) | +0.40 (-0.15) | +0.50 (+0.09) | +0.22 (+0.16) | +0.36 (+0.17) |
| control | `acc_var_init_randlab`: Var_k of 0-1 risk, permuted labels | -0.19 (-0.07) | -0.00 (-0.04) | +0.12 (+0.19) | +0.16 (+0.20) | +0.09 (+0.15) |
| control | `var_init_norm_randlab`: Var_k of CE risk (norm.), permuted labels | -0.11 (+0.04) | -0.17 (+0.02) | +0.09 (+0.32) | +0.21 (+0.30) | +0.06 (+0.21) |
| baseline | `cvu`: CVU (Gracheva 2021) = std/mean of untrained acc | +0.37 | +0.46 | +0.49 | +0.19 | +0.34 |
| baseline | `zico`: ZiCo (simplified) | -0.36 (-0.26) | -0.47 (-0.36) | +0.35 (+0.29) | -0.20 (-0.15) | -0.30 (-0.21) |
| baseline | `log_n_params`: log #params | +0.16 (+0.27) | +0.07 (+0.19) | — | +0.19 (+0.20) | +0.23 (+0.26) |

- digits, full: n=1200, test acc median 0.955, range 0.081-0.980, seed std (median) 0.0048
- digits, n=200: n=1200, test acc median 0.869, range 0.085-0.936, seed std (median) 0.0100
- digits, gain sweep: n=120, test acc median 0.934, range 0.081-0.973, seed std (median) 0.0055
- Fashion, n=10k: n=1200, test acc median 0.829, range 0.095-0.856, seed std (median) 0.0038
- Fashion, n=1k: n=1200, test acc median 0.766, range 0.097-0.799, seed std (median) 0.0037
