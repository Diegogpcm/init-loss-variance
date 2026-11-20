# E-dp: certificates for a single network trained with DP-SGD (mutual information + Markov), theory.md §14

δ = 0.025 split as δ₁ = 0.024 (Markov on the information density) and δ₂ = 0.001 (disintegrated PAC-Bayes-kl).
I ≤ n·T·KL_step(q, σ) (Poisson-subsampled Gaussian, per-example chain rule). ν = noise std on the mean gradient (units
of the clip C = 1); T = number of steps. "no-Markov" plugs E[ı] in place of its tail and is **not** a certificate.

## Comparison on MNIST (n = 60,000, standard test set), data-free / no held-out data

| method | model | certified predictor | test error | **certificate** |
|---|---|---|---|---|
| **ours: DP-SGD + MI (§14), recipe A, chosen on Fashion, 3 seeds** | linear softmax on data-free Gabor features (8,340 params) | the trained network | 0.058 ± 0.004 | **0.099 ± 0.004** |
| ours: DP-SGD + MI, recipe B (3 seeds) | 1-hidden-layer LN MLP (128) on Gabor features | the trained network | 0.081 ± 0.006 | 0.116 ± 0.003 |
| ours: Gaussian PAC-Bayes, bn_stat + inv KL + klinv01 (3 seeds) | MLP 784-64-64-10, BN stats as parameters | Gibbs (stochastic) | 0.067 ± 0.003 | 0.228 ± 0.002 |
| ours: same, plain network (inv KL, klinv01) | MLP 784-64-64-10 | Gibbs | 0.080 ± 0.003 | 0.267 ± 0.004 |
| baseline in our pipeline: Pérez-Ortiz-style (plain KL, f_quad) | MLP 784-64-64-10 | Gibbs | 0.203 ± 0.018 | 0.349 ± 0.021 |
| García-Pérez et al. 2025, data-free prior, best (KL-modulated, 0-1 objective) | MLP 3×600 | Gibbs | — | 0.315 |
| Pérez-Ortiz et al. 2021 (JMLR), random-init prior, f_quad | FCN / CNN | Gibbs | — | ≈0.32 / 0.2165 |
| Lampert & Zakerinia 2026, DP-SGD prior + Gaussian posterior | MLP | Gibbs | — | 0.29 (data-free prior 0.32) |
| Lotfi et al. 2022, compression prior | compressed model (~380 bytes) | the compressed network | — | 0.116 |
| Pérez-Ortiz et al. 2021, prior learned on half the data (uses a split) | CNN | Gibbs | ~0.01 | 0.0155 |

Published numbers are from docs/literature.md (as reported by the papers; not rerun). Dziugaite & Roy (2017) report
0.161 on *binary* MNIST (not comparable).

## CIFAR-10 (n = 50,000, standard test set), same recipes transferred unchanged (Gabor map applied per RGB channel)

| method | certified predictor | test error | **certificate** |
|---|---|---|---|
| ours: DP-SGD + MI, recipe A (linear on Gabor, 32,650 params), 3 seeds | the trained network | 0.675 ± 0.003 | **0.740 ± 0.005** |
| ours: recipe B (1-hidden-layer MLP on Gabor), 3 seeds | the trained network | 0.715 ± 0.009 | 0.761 ± 0.006 |
| García-Pérez et al. 2025, data-free prior, best (4-layer CNN, 9,946 params, f̃_mb) | Gibbs | — | 0.738 ± 0.007 |
| Lampert & Zakerinia 2026, DP-SGD prior (deep ConvNet) | Gibbs | — | 0.84 |

A tie with the best data-free CNN certificate, from weak first-order features (no tuning on CIFAR). Better data-free
features (second-order scattering, colour handling) are the obvious lever and would need a pre-registered choice.

## Fashion-MNIST (n = 55,000), recipes and re-seeding

Recipe A: linear softmax on Gabor features (zero init), B = 4096 (Poisson), ν = 0.01, 10 epochs (T = 134 on Fashion,
146 on MNIST), SGD momentum 0.9, lr 0.3, cosine decay → I ≈ 12.2 nats.
Recipe B: 1-hidden-layer LN MLP (width 128) on Gabor features, ν = 0.01, 5 epochs, lr 1 → I ≈ 6.1 nats.

| dataset | recipe | seeds | train | test | I (nats) | **cert** | no-Markov |
|---|---|---|---|---|---|---|---|
| Fashion | A | 3 | 0.213 ± 0.002 | 0.210 ± 0.004 | 12.2 | **0.273 ± 0.003** | 0.224 |
| Fashion | B | 3 | 0.241 ± 0.011 | 0.238 ± 0.011 | 6.1 | 0.286 ± 0.011 | 0.252 |
| MNIST | A | 3 | 0.062 ± 0.003 | 0.058 ± 0.004 | 12.2 | **0.099 ± 0.004** | 0.069 |
| MNIST | B | 3 | 0.086 ± 0.003 | 0.081 ± 0.006 | 6.1 | 0.116 ± 0.003 | 0.093 |

Fashion seed 0 of each recipe comes from the selection grid (round 3); seeds 1–2 are fresh. On Fashion the recipe was
selected from ~120 runs on the same S, so Fashion numbers carry an unpaid selection cost; MNIST numbers do not.
Best previous Fashion certificate (Gaussian PAC-Bayes): 0.347 ± 0.001.

## Selection grid on Fashion (rounds 1–3), best 20 single runs

| file | features | depth | norm | width | params | B | ν | T | lr | sched | train | test | I ≤ (nats) | **cert** | no-Markov (not a cert.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.01 | 134 | 0.3 | cos | 0.2137 | 0.2055 | 12.2 | **0.2745** | 0.2255 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 128 | 108298 | 4096 | 0.01 | 67 | 1.0 | cos | 0.2319 | 0.2245 | 6.1 | **0.2764** | 0.2422 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 128 | 108298 | 4096 | 0.01 | 134 | 0.3 | cos | 0.2194 | 0.2089 | 12.2 | **0.2806** | 0.2313 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.01 | 269 | 0.3 | cos | 0.2001 | 0.1977 | 24.5 | **0.2851** | 0.2145 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 32 | 27082 | 4096 | 0.01 | 134 | 1.0 | cos | 0.2245 | 0.2212 | 12.2 | **0.2861** | 0.2365 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 128 | 108298 | 4096 | 0.01 | 134 | 1.0 | cos | 0.2248 | 0.2198 | 12.2 | **0.2864** | 0.2367 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 32 | 27082 | 4096 | 0.01 | 134 | 0.3 | cos | 0.2266 | 0.2250 | 12.2 | **0.2883** | 0.2386 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.01 | 134 | 1.0 | cos | 0.2288 | 0.2232 | 12.2 | **0.2907** | 0.2408 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.02 | 134 | 0.3 | cos | 0.2575 | 0.2496 | 3.0 | **0.2912** | 0.2670 |
| dp_sgd_r3_pix_lin | pixels | 0 | ln | — | 7850 | 4096 | 0.01 | 67 | 0.3 | cos | 0.2463 | 0.2385 | 6.1 | **0.2916** | 0.2568 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.01 | 67 | 0.3 | cos | 0.2471 | 0.2424 | 6.1 | **0.2924** | 0.2576 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.005 | 67 | 1.0 | cos | 0.2069 | 0.2052 | 24.4 | **0.2926** | 0.2214 |
| dp_sgd_r3_pix_lin | pixels | 0 | ln | — | 7850 | 4096 | 0.01 | 134 | 0.3 | cos | 0.2315 | 0.2271 | 12.2 | **0.2936** | 0.2436 |
| dp_sgd_r3_pix_lin | pixels | 0 | ln | — | 7850 | 4096 | 0.005 | 27 | 1.0 | cos | 0.2385 | 0.2338 | 9.8 | **0.2949** | 0.2501 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 128 | 108298 | 4096 | 0.005 | 67 | 1.0 | cos | 0.2099 | 0.2021 | 24.4 | **0.2959** | 0.2245 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 128 | 108298 | 4096 | 0.01 | 67 | 0.3 | cos | 0.2508 | 0.2447 | 6.1 | **0.2963** | 0.2613 |
| dp_sgd_r3_gabor_mlp | gabor | 1 | ln | 32 | 27082 | 4096 | 0.01 | 67 | 1.0 | cos | 0.2543 | 0.2513 | 6.1 | **0.2999** | 0.2648 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.02 | 269 | 0.3 | cos | 0.2546 | 0.2491 | 6.1 | **0.3004** | 0.2652 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.01 | 67 | 1.0 | cos | 0.2549 | 0.2514 | 6.1 | **0.3006** | 0.2655 |
| dp_sgd_r3_gabor_lin | gabor | 0 | ln | — | 8340 | 4096 | 0.005 | 27 | 3.0 | cos | 0.2440 | 0.2416 | 9.8 | **0.3007** | 0.2556 |

## Tail by Lampert, Cairney-Leeming & Zakerinia (2026, v2) Thm 8 instead of Markov (2026-11-20)

Same runs, rescored with `experiments/rescore_tail.py`; κ depends only on (n, q, σ, T), with β = 0.024 and δ₂ = 0.001.
Recipe A: κ = 185 (Markov 522). Recipe B: κ = 104 (Markov 269).

| dataset (recipe A unless noted) | seeds | Markov | **Thm 8** | test |
|---|---|---|---|---|
| Fashion-MNIST (tuning seed 0 + seeds 1, 2) | 3 | 0.273 ± 0.003 | **0.249 ± 0.003** | 0.210 |
| MNIST (pre-registered transfer) | 3 | 0.0995 ± 0.0050 | **0.0836 ± 0.0047** | 0.058 |
| CIFAR-10 (pre-registered transfer) | 3 | 0.740 ± 0.006 | **0.715 ± 0.006** | 0.675 |
| MNIST, recipe B | 3 | 0.116 ± 0.004 | 0.105 ± 0.004 | 0.081 |
| CIFAR-10, recipe B | 3 | 0.761 ± 0.007 | 0.745 ± 0.007 | 0.716 |

Re-ranking the Fashion tuning grid under Thm 8: the top of the grid is flat. Configurations within 0.003 of each other
are ν = 0.01 with 10 or 20 epochs and ν = 0.005 with 10 epochs, lr 1. Recipe A stays at the top within seed noise, so
the pre-registered selection is unchanged.

Ceiling at fixed hyperparameters (MNIST A, seed-mean train error):

| κ | value | certificate |
|---|---|---|
| Thm 8 | 185 | 0.084 |
| 2I + 2√(2IL) | 43 | 0.073 |
| I + 2√(IL) | 26 | 0.071 |
| E[ı], not a certificate | 12 | 0.069 |
