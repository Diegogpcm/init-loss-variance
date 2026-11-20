# Init-loss variance as an architecture-quality proxy

**Question.** Does the variance of the loss at initialization, with the
initialization distribution $P_0$ treated as a PAC-Bayes prior, tell us how
good an architecture will be after training?

This repo has a NumPy-only testbed (MLPs with manual backprop, sklearn `digits`
dataset, no downloads), a set of loss-variance proxies with literature baselines,
and three initial experiments. Status: **exploratory**. Detailed, paper-oriented documentation lives in [`docs/`](docs/): [theory](docs/theory.md), [literature](docs/literature.md), [experiments](docs/experiments.md), [research log](docs/research_log.md), [paper outline](docs/paper_outline.md). v2 adds full-grid runs, accuracy (0-1 loss) variance, and Fashion-MNIST ([below](#v2-full-grids-accuracy-0-1-loss-variance-harder-data)).

## Why this might work: a PAC-Bayes motivation

Let $\hat L(\theta)$ be the empirical risk on $n$ examples and $\theta\sim P_0$ the
init distribution. The minimum over posteriors $\rho$ of the Catoni-style objective
$\mathbb E_\rho \hat L + \mathrm{KL}(\rho\|P_0)/(\lambda n)$ is attained by the Gibbs
posterior. Its value is the **free energy**

$$F(\lambda) = -\tfrac{1}{\lambda n}\log \mathbb E_{\theta\sim P_0}\, e^{-\lambda n \hat L(\theta)}
\;=\; \mu_0 - \tfrac{\lambda n}{2}\,\sigma_0^2 + O(\kappa_3),$$

where $\mu_0,\sigma_0^2$ are the **mean and between-init variance of the
empirical risk under the prior**. So at a fixed mean init loss, *larger* variance
means more prior mass at low loss, which means a tighter bound. This is the
opposite direction from the "stable architectures are better" intuition in the
earlier literature (below). Both quantities can be estimated with forward passes
only.

The law of total variance splits the loss variance into a between-init part
(`var_init`, the one PAC-Bayes cares about) and a within-init, across-example part
(`var_data`). The two can behave very differently.

## Related work and novelty

| Work | What it measures at init | Relation |
|---|---|---|
| Gracheva 2021, *Array* ("CVU") | CV of **untrained accuracy** over many random inits; lower CV reported as better on NAS-Bench-201 | Closest prior work. Same idea (dispersion over inits), but uses accuracy and no PAC-Bayes link |
| Gracheva 2023/25, *epsinas* | Normalized dispersion of raw **outputs** under two constant shared-weight inits | Output-level, not loss-level |
| Bowden et al. 2021 (Caltech course report) | PAC-Bayes-inspired: average label-free triplet loss over 100 inits | Mean, not variance; unpublished |
| Abdelfattah et al. 2021 (zero-cost proxies), Mellor et al. 2021 (NASWOT), Li et al. 2023 (ZiCo), Týbl & Neumann CVPR 2025 (Fisher variance) | Gradient or activation statistics, mostly at **one** init | Baselines; ZiCo and L-SWAG use *gradient* variance, not loss variance |
| Ru et al. 2021 (SoTL), Lotfi et al. 2022 | Marginal likelihood or training-speed estimators | Same Bayesian motivation, but they need training |
| **Valle-Pérez, Camargo & Louis 2019 (ICLR)** | Apply the existing realizable PAC-Bayes bound (Langford & Seeger 2001 / McAllester 1998) with the *function-space* prior induced by random parameters. The bound is driven by $-\ln P_0(\text{zero training error})/m$ (the marginal likelihood), computed with an NNGP + EP approximation. Binarized MNIST/Fashion/CIFAR, 2 architectures, label corruption; the appendix shows the bound is best near the order/chaos transition in weight variance | **Closest PAC-Bayes prior work.** Same prior and 0-1 loss, in the realizable ($\lambda\to\infty$) limit. No new bound. No variance/cumulant view, no MC over inits of the risk itself |
| **Valle-Pérez & Louis 2020** | New (minor) variant, Thm 5.1: the same bound holding w.h.p. over posterior samples, not in expectation. Thm 6.4: it is asymptotically optimal up to a constant when learning curves are power laws. Kernel estimated by MC over $M$ random inits, then EP. 19 architectures × 5 datasets × several $m$; bound correlates with error across architectures, but differences are small | Lists NAS as future work. Requires binary labels, zero training error, and a Gaussian approximation to the prior |
| Park et al. 2020 (arXiv:2011.06006) | NNGP performance (kernel by MC over finite nets at init) as a cheap NAS signal on NAS-Bench-101 and MNAS | NAS with the init-induced prior, but via GP prediction, without PAC-Bayes |
| Mingard et al. 2021 (JMLR) | SGD's function distribution ≈ Bayesian posterior under the init prior | Justifies using the init prior to reason about SGD-trained nets |
| Germain et al. 2016 (NeurIPS) | PAC-Bayes bound with NLL loss ⇔ Bayesian marginal likelihood | The CE free energy at λ=1 is a log-evidence |
| **Picard-Weibel et al. 2025 (arXiv:2503.08231)** | No new bound, no experiments. They prove that (i) for any bound of the form $b(\pi[R], \pi_p[D(d\pi/d\pi_p)])$ (Catoni, McAllester, Maurer, …), the infimum over posteriors depends only on the push-forward $\pi_p^{\#R}$ (the "risk prior"); (ii) that infimum is monotone in the stochastic order; (iii) for Catoni, a guarantee $G$ needs prior mass $\gtrsim e^{-2G^2 n}/\delta$ on risk $\le G/(1-\alpha)$. They argue this makes PAC-Bayes *witness* rather than explain generalization | Formal version of our premise. Their order result says nothing about distributions with **equal means** (all incomparable), so the variance-at-fixed-mean argument is not covered |

**Assessment: partially novel.** Using dispersion over random inits as a proxy is
not new (Gracheva's CVU), and using the init distribution as a PAC-Bayes prior to explain
why some architectures generalize is not new either (Valle-Pérez et al. 2019, 2020, who
bound the error with the prior mass of zero-training-error functions). As far as I can find,
three things are new:
(i) the specific object, the *loss* variance and its between-init vs. within-init
decomposition; (ii) the free-energy / cumulant framing, which predicts the *sign*
of the effect; (iii) temperature normalization, which separates function shape
from logit scale. The main risk is that (iii) collapses back into CVU (see results).

## Proxies (`lvproxy/proxies.py`)

Computed from $K=32$ inits on a fixed probe batch ($B=256$, or 200 in the low-data run).

- `var_init`: $\mathrm{Var}_k[\hat L(\theta_k)]$, the between-init variance of the empirical risk
- `var_data`: $\mathbb E_k \mathrm{Var}_i[\ell_{ik}]$, the across-example variance within an init
- `var_pointwise`: $\mathbb E_i \mathrm{Var}_k[\ell_{ik}]$
- `free_energy_λ`: Monte Carlo estimate of $F(\lambda)$; `cumulant_λ`: its 2nd-order expansion
- `*_norm`: the same quantities after rescaling each init's (per-example-centred) logits to unit std, which removes logit scale
- Baselines: `cvu`, `naswot`, `grad_norm`, `zico` (simplified, 4 minibatches), `log_n_params`, `untrained_acc`

## Experiments and first results

All results use test accuracy averaged over 3 training seeds (Adam, lr 1e-3, grad clipping).
Search space: depth ∈ {1,2,4,8,12}, width ∈ {16,…,128}, activation ∈ {relu, tanh,
sigmoid, linear, gelu, sin}, residual ∈ {yes,no}, init gain ∈ {0.5,…,3}.

| Experiment | Script | Setup |
|---|---|---|
| E1 | `run_sweep.py` | 150 random archs, full digits (1347 train), 40 epochs |
| E2 | `run_sweep.py --n-train 200 --epochs 150` | same 150 archs, 200 training examples |
| E3 | `run_gain_sweep.py` | depth 8, width 64; 6 activations × {plain,res} × 10 gains (0.25–4) |

![correlations](figures/proxy_correlations.png)

Spearman ρ with test accuracy (full tables, including Kendall τ, the acc>0.5 subset and
within-activation correlations, are in [`results/correlation_tables.md`](results/correlation_tables.md)):

| proxy | E1 full | E2 low-data | E3 gain sweep |
|---|---|---|---|
| `var_init` (raw) | −0.09 | −0.11 | +0.14 |
| `var_init` raw, partial on mean init loss | **+0.30** | **+0.26** | **+0.47** |
| `var_init_norm` | **+0.38** [0.23, 0.51] | **+0.43** [0.28, 0.55] | +0.25 [0.07, 0.41] |
| `free_energy_1_norm` (lower = better) | −0.26 | −0.38 | −0.20 |
| `var_data` (raw) | −0.17 | −0.18 | — |
| CVU (Gracheva) | +0.48 | +0.56 | +0.49 |
| ZiCo (simplified) | −0.37 | −0.46 | +0.35 |
| grad norm | −0.32 | −0.36 | −0.15 |
| log #params | +0.13 | +0.05 | — |

### What we learned so far

1. **Raw loss variance is dominated by logit scale.** On its own it is uninformative
   (|ρ| ≈ 0.1) because large-gain networks have huge, wildly varying initial losses.
   Holding the mean init loss fixed (partial correlation) brings out a consistent
   **positive** effect (+0.26 to +0.47). That is the sign the free-energy argument predicts.
2. **Temperature-normalized between-init variance works**, with ρ ≈ +0.4 across both
   sweeps. Within-init / across-example variance (`var_data`, `var_pointwise_norm`)
   points the *other* way. The decomposition matters.
3. **But it is largely redundant with CVU.** `var_init_norm` and CVU have rank
   correlation 0.73–0.82. Controlling for CVU leaves `var_init_norm` no extra signal
   (partial ρ −0.03 / −0.04 / −0.18), while CVU keeps +0.32 to +0.46 beyond it.
   In this MLP setting, **higher** CVU predicts **better** accuracy, the opposite
   sign from Gracheva's NAS-Bench-201 finding. That needs checking in her setting.
4. **Mostly a trainability detector.** The signal comes mainly from separating
   degenerate priors (input-insensitive or chaotic random functions: deep sin,
   sigmoid, high-gain tanh) from working ones. Among architectures with acc > 0.5,
   ρ drops to about +0.2.
5. **Gain sweep (E3):** within each activation, normalized variance tracks the
   ordered-to-chaotic transition for sin, tanh and gelu (within-activation ρ = +0.47).
   It misses plain sigmoid recovering at high gain. By construction it cannot see
   ReLU gain changes, because ReLU is homogeneous and normalization cancels the scale.

![gain sweep](figures/gain_sweep.png)

![scatter](figures/scatter_n200.png)

### Caveats

- `digits` is easy (median test accuracy about 0.95, seed std about 0.006), so rankings at the top are near noise.
- MLPs only. No CNNs, BatchNorm or NAS benchmarks yet. ZiCo here is a simplified re-implementation, and its sign is unstable across experiments.
- The free-energy MC estimate with $K=32$ is dominated by the minimum when $\lambda n$ is large.

## v2: full grids, accuracy (0-1 loss) variance, harder data

Run on kipchoge (72 CPUs) with `bash experiments/run_v2.sh digits` and `... fashion`
(~45 min total). Changes from v1:

- **Full architecture grid** (all 1200 configs) instead of 150 random ones, $K=128$ inits instead of 32.
- **Fashion-MNIST** (fixed stratified 10k train / 5k test subset, plus a 1k-train low-data run), where
  the top of the ranking is no longer saturated.
- **0-1 loss (accuracy) family**, computed from the per-example error matrix $e_{ik}$ under the same inits:
  `acc_var_init` $=\mathrm{Var}_k[\text{err}_k]$, `acc_var_pointwise`, `acc_overdispersion`
  (between-init variance divided by its value if examples were flipped independently,
  $=1+(B-1)\bar\rho$), the 0-1 free energy `acc_free_energy_λ`, and `acc_best` (best-of-K, the $\lambda\to\infty$ limit).
  The 0-1 loss is bounded, so it is the loss for which the Catoni bound holds as stated; CE is not.
- **Controls:** the same quantities with probe labels permuted (`*_randlab`), and the CE free energy
  minimized over a temperature grid (`free_energy_opt_λ`, next step 3 below).

Spearman ρ with mean test accuracy (3 seeds); in parentheses, partial ρ controlling for CVU.
Full tables: [`results/v2/summary.md`](results/v2/summary.md), [`tables_digits.md`](results/v2/tables_digits.md),
[`tables_fashion.md`](results/v2/tables_fashion.md).

| proxy | digits full | digits n=200 | gain sweep | Fashion 10k | Fashion 1k |
|---|---|---|---|---|---|
| `var_init_norm` (CE, normalized) | +0.26 (−0.10) | +0.33 (−0.10) | +0.32 (−0.25) | +0.04 (−0.23) | +0.21 (−0.16) |
| `free_energy_opt_1` (CE, best T; lower=better) | −0.22 | −0.28 | −0.26 | −0.04 | −0.22 |
| **`acc_var_init`** (0-1) | **+0.32** (−0.17) | **+0.41** (−0.16) | **+0.50** (+0.08) | **+0.22** (+0.16) | **+0.36** (+0.16) |
| `acc_overdispersion` | +0.34 (−0.19) | +0.43 (−0.18) | +0.49 (−0.03) | +0.20 (+0.13) | +0.35 (+0.13) |
| **`acc_best`** / 0-1 free energy λ≥1 | +0.32 (+0.05) | +0.39 (+0.06) | **+0.57 (+0.34)** | **+0.24** (+0.15) | **+0.36** (+0.16) |
| `untrained_acc` (mean) | −0.28 | −0.25 | +0.30 | +0.09 | +0.07 |
| `acc_var_init_randlab` (permuted labels) | −0.19 | −0.00 | +0.12 | +0.16 | +0.09 |
| CVU (Gracheva) | +0.37 | +0.46 | +0.49 | +0.19 | +0.34 |
| ZiCo (simplified) | −0.36 | −0.47 | +0.35 | −0.20 | −0.30 |
| log #params | +0.16 | +0.07 | — | +0.19 | +0.23 |

Within-activation ρ is higher across the board (+0.36 to +0.52 for `acc_var_init`).

### What v2 adds

1. **Accuracy variance beats CE-loss variance everywhere.** `acc_var_init` is higher than
   `var_init_norm` in all five experiments, and it needs no temperature normalization, because
   argmax ignores logit scale. Temperature-optimizing the CE free energy does not rescue the CE version.
2. **CVU ≈ accuracy variance, and the sign is the one PAC-Bayes predicts.** CVU is
   $\sqrt{\texttt{acc\_var\_init}}/\texttt{untrained\_acc}$. On digits the mean untrained accuracy is
   *negatively* correlated with final accuracy, so dividing by it helps and CVU edges out raw variance.
   On Fashion that correlation is near zero and the raw variance wins, keeping partial ρ ≈ +0.16 beyond CVU.
   The CE variance has *no* signal left after CVU on any dataset.
3. **The lower tail matters more than the variance.** `acc_best` (equivalently, the 0-1 free energy at
   λ≥1, which is dominated by the best inits) is the best or tied-best proxy in 4 of 5 experiments, and
   on the gain sweep it keeps partial ρ = +0.34 beyond CVU. This matches the large-λn regime of the PAC-Bayes
   argument, where the free energy depends on the prior mass of *low-error* functions, not on the second cumulant.
4. **The signal is mostly label-dependent.** With permuted labels, error overdispersion drops to ≈1
   (errors become independent across examples), and on digits the variance loses its predictive power
   (ρ −0.19 / 0.00). On Fashion a smaller label-independent part remains (+0.16 / +0.09).
5. **Variance and tail are tightly linked for the 0-1 loss.** Because output units are exchangeable at init,
   E[untrained acc] = 1/C exactly; it is 0.100 within sampling error for ~99% of architectures.
   With the mean pinned near the boundary, extra variance mostly comes from the upper accuracy tail:
   ρ(best-of-128, std) ≈ 0.80 in every experiment, and the best init sits a median 3.0 SD above the mean
   (IQR 2.7–3.5, slightly heavier than a Gaussian's ≈2.6). The leftover shape, (best − mean)/std, still predicts
   test accuracy (ρ +0.12 to +0.41), so tail and variance are related but not interchangeable.
6. **Still modest.** |ρ| is 0.2–0.57, about the same as log #params on Fashion. Among working architectures
   (acc > 0.5) it drops to 0.12–0.37. The trainability-detector caveat from v1 still applies.

### Can the prior's mean and variance give a better bound? (theory note)

Not as an absolute bound. Optimized over posteriors, Catoni/Maurer already depend on the *whole* empirical-risk
distribution under the prior (Picard-Weibel et al. 2025, Thm 1). Knowing only (μ, σ²) yields a *worst case* over all
distributions with those moments ([`experiments/moment_worst_case.py`](experiments/moment_worst_case.py), exact
enumeration of 3-atom extreme points). With μ = 0.9 pinned by symmetry and δ = 0.05:

| n | σ = 0.01 | 0.03 | **0.04** (observed median) | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| 1347 | 0.932 | 0.925 | **0.918** | 0.910 | 0.837 | 0.541 |
| 60000 | 0.901 | 0.895 | **0.886** | 0.880 | 0.806 | 0.506 |

At the observed σ ≈ 0.04 this is worse than the 0.9 that the symmetry gives for free. The worst case piles mass just
below the mean (≈ μ − σ²/(1−μ)) plus an atom at risk 1, so moments cannot certify a low-risk tail. The worst-case bound
is, however, monotone in σ at fixed μ. That makes it a minimax justification for ranking architectures by variance,
not a usable certificate. The statistic a bound actually needs is the MGF $\mathbb E_{P_0}e^{-\hat R/\lambda}$, i.e. the free energy.

### Parameter-space vs function-space KL (Fashion-MNIST 60k, PAC-Bayes-trained Gaussian posteriors)

[`experiments/function_space_kl.py`](experiments/function_space_kl.py) (PyTorch). The prior is N(m₀, σ₀²) with a random
data-free m₀, as in Pérez-Ortiz et al. and García-Pérez et al. The posterior is a diagonal Gaussian trained for 50
epochs on a McAllester-style objective. KL is measured on logits at m training inputs, softmax-shift removed. By data
processing, KL_m ≤ KL_function ≤ KL_param. The *linear* estimate is an exact push-forward through a shared Jacobian,
so it can never exceed KL_param.

Share of KL_param visible through m inputs (linear model; the 32-wide nets, where m reaches 160):

| | m = 10 | m = 40 | m = 160 (1,440 output dims vs ~27k params) |
|---|---|---|---|
| σ₀ = 0.02 | 28–46% | 54–70% | **75–85%**, still rising |
| σ₀ = 0.05 | 19–29% | 39–48% | 58–63%, still rising |

- With 160 inputs, most of the parameter KL is already visible in function space. The share is higher at smaller σ₀,
  as expected: near the linear regime the two KLs differ only along exact invariances.
- 92–99% of KL_param is the mean term ‖μ − m₀‖²/2σ₀².
- Certificates are 0.40–0.55 (test error of Q 0.29–0.44). The data-free prior is the binding constraint here.
- The Monte Carlo Gaussian fit (no linearization) is **not reliable**. It exceeds KL_param for d2-w32-relu at σ₀ = 0.02
  (3,551 > 2,537), which is impossible, so the logits are too non-Gaussian for a Gaussian fit. It is kept in the
  results file, not used for conclusions.
- Conclusion: for Gaussian posteriors at the σ that gives good bounds, re-measuring the KL in function space saves
  little. Gains must come from the prior and from the posterior family, not from where the KL is measured.

### Normalization in PAC-Bayes: BatchNorm with statistics as parameters, and invariance-aware KL

[`experiments/pacbayes_norm.py`](experiments/pacbayes_norm.py) (PyTorch). Setup: Fashion-MNIST with n = 55k, depth-2
ReLU MLPs, a data-free prior N(m₀, σ₀²), and a Gaussian posterior trained 50 epochs. The certificate is PAC-Bayes-kl
with δ = 0.025, plus the Langford–Caruana correction for estimating the Gibbs error from M = 150 posterior samples
(δ' = 0.01). Full tables: [`results/v2/norm_summary.md`](results/v2/norm_summary.md).

**Making BatchNorm i.i.d.-compatible.** Training-mode BN couples the examples in a batch, and inference-mode running
statistics make the weights → classifier map depend on S. Workarounds:
- `bn_stat` (preferred): the inference-mode statistics (m_j, log s_j) are **posterior parameters** with a data-free
  prior N(0, 0.5²). Phase 1 trains with batch statistics. Phase 2 freezes them into the posterior (trainable mean and
  sd) and fine-tunes in inference mode, so the statistics see the weight noise. The classifier is a per-example network,
  and (x·W_j − m_j)/s_j is *exactly* invariant to scaling (W_j, m_j, s_j) by c > 0.
- `bn_param`: naive version, with statistics frozen after training and a fixed sd. Kept as a baseline.
- `bn_ref`: statistics recomputed for each sampled weight vector on a held-out *unlabeled* reference set (virtual BN).
  It works, but needs held-out inputs.

**Invariance-aware KL.** If rescaling the weights leaves every classifier unchanged, Q and its rescaled Q_t give the same
Gibbs classifier, so the certificate may use min_t KL(Q_t‖P). The invariances used: ReLU per-unit homogeneity,
LN per-layer scale, BN per-unit scale. Empirical risk is always evaluated with Q_t's own samples. `inv` trains t jointly
with (μ, σ); "post-hoc" minimizes over t after plain training.

Certificate (and KL, nats). `plain` → `inv`:

| σ₀ | width | none | LayerNorm | BN, reference set | BN stats naive | **BN stats as params** |
|---|---|---|---|---|---|---|
| 0.02 | 64 | 0.627 → 0.568 (2294 → 1540) | 0.628 → 0.569 (2079 → 1097) | 0.607 → 0.559 (2238 → 1038) | 0.620 (2611) | 0.599 → **0.529** (1761 → **775**) |
| 0.02 | 128 | 0.619 → 0.610 (2835 → 1922) | 0.626 → 0.561 (2729 → 1242) | 0.632 → 0.558 (3227 → 1136) | 0.628 (4162) | 0.595 → **0.515** (2378 → **949**) |
| 0.05 | 64 | 0.553 → 0.521 (1753 → 1162) | 0.567 → 0.518 (1528 → 1018) | 0.548 → **0.514** (1663 → 1060) | 0.586 (2335) | 0.541 → 0.519 (1402 → **803**) |
| 0.05 | 128 | 0.641 → 0.570 (3192 → 2024) | 0.606 → 0.537 (2257 → 1153) | 0.573 → 0.527 (2580 → 1169) | 0.618 (3513) | 0.580 → 0.537 (2386 → 1112) |

Ordinary training (Adam, 20 epochs, 3 seeds), test error at widths 64 / 128: none 0.113 / 0.111, LN 0.115 / 0.111,
**BN 0.111 / 0.108**, BN with a reference set 0.115 / 0.108. BN is slightly but consistently best.

- **BN with statistics as parameters, trained with the invariance-aware KL, gives the lowest KL in every setting and the
  best certificate at σ₀ = 0.02** (0.515 at width 128). It needs no held-out data, and its frozen statistics cost only
  about 220–360 nats (plain runs). The naive version is the worst BN variant.
- **Training with the invariance-aware KL** cuts the KL by 30–65% and the certificate by up to 0.08, for every
  normalization including none (ReLU homogeneity). **Post-hoc rescaling saves under 1%**. The gain comes from giving the
  optimizer the rescalings as free parameters (a better path to a lower-KL posterior), not from removing nats wasted
  along invariant directions.
- With rescaling, normalized networks certify better than plain ones at width 128 (0.52–0.54 vs 0.57–0.61). At width 64
  the differences are mostly within noise.
- The certificates here include the Monte Carlo sampling correction, so they are not comparable to the function-space
  KL runs above, which omit it.
- Caveat: in the first `bn_stat` batch, the per-run "statistics KL" of `inv` runs was computed on unrescaled parameters
  (fixed in the code since). Total KLs and certificates are unaffected.

### Certifying a network trained with noisy SGD: mutual information instead of a Gaussian posterior

[`experiments/dp_sgd_bound.py`](experiments/dp_sgd_bound.py), derivation in [`docs/theory.md`](docs/theory.md) §14,
tables in [`results/v2/dp_summary.md`](results/v2/dp_summary.md).

The certified predictor is the network output by noisy, clipped, Poisson-subsampled SGD (DP-SGD). The prior is the
algorithm's (unknown, data-free) output marginal. A disintegrated PAC-Bayes-kl bound charges the information density,
whose tail is bounded with high probability by Thm 8 of Lampert, Cairney-Leeming & Zakerinia (2026, v2) from replace-one
Rényi DP with Poisson amplification (earlier: Markov, I(S; W)/δ₁). The mutual information is bounded with the 1/n per example:
I ≤ n·T·KL(subsampled Gaussian) ≈ T/(2ν²n), where ν is the noise std on the mean gradient. SGD noise is what makes
reusing the data across epochs cheap, and it is now part of the certificate.

| setting (data-free, no held-out data) | certificate |
|---|---|
| Fashion-MNIST, best Gaussian PAC-Bayes (bn_stat + inv KL + klinv01) | 0.347 ± 0.001 |
| Fashion-MNIST, DP-SGD, linear on data-free Gabor features (3 seeds): tail by L-CL-Z Thm 8 / by Markov | **0.249 ± 0.003** / 0.273 ± 0.003 (test 0.210) |
| MNIST, same recipe transferred unchanged (3 seeds): Thm 8 / Markov | **0.084 ± 0.005** / 0.099 ± 0.004 (test 0.058) |
| MNIST, our Gaussian PAC-Bayes pipeline (bn_stat MLP) / Pérez-Ortiz-style f_quad baseline | 0.228 / 0.349 |
| CIFAR-10, same recipe unchanged (3 seeds): Thm 8 / Markov | **0.715 ± 0.006** / 0.740 ± 0.005 (test 0.675) |
| CIFAR-10, published: García-Pérez 2025 best data-free CNN; L-CL-Z 2026 v2 best (DP-SGD prior + posterior, 4-layer CNN) | 0.738 ± 0.007; 0.716 ± 0.007 |
| MNIST, published: García-Pérez 2025 (MLP 3×600), Pérez-Ortiz 2021 (CNN), L-CL-Z 2026 v2 (DP-SGD prior: posterior / network), Lotfi 2022 (compression) | 0.315 / 0.2165 / 0.233 / 0.248 / 0.116 |

- The optimum is a very noisy, short run (I ≈ 6–25 nats, i.e. ~10⁻⁴ nats per example). Small models (linear on
  data-free features) beat MLPs, as in DP practice (Tramèr & Boneh 2021). The generalization gap is ≈ 0; the slack is
  the tail of the information density (κ = 185 nats with Thm 8 against E[ı] ≤ 12; the not-a-certificate plug-in E[ı]
  gives 0.224 on Fashion and 0.069 on MNIST) plus training error under noise.
- Remaining lever (theory.md §14.1): Thm 8 still pays ≈ ln(4/β₁) for the realised posterior divergences, plus a factor
  ≈ 2 for replace-one. A structured Gaussian-chaos tail (theory.md §14.2) is worth ≤ 0.008 at the measured gradient
  diversity (effective rank 9–21), so it was dropped. The bigger slack is worst-case sensitivity: centred gradient
  variance is 0.37–0.78 of the antipodal worst case's scale, but individual Rényi-filter accounting does not convert it
  into better certificates: the filter drops the misclassified points first (§14.3).

## Next steps (proposed)

1. **Move to PyTorch and NAS-Bench-201 / NATS-Bench** (the main literature benchmark) so the numbers compare directly with CVU, ZiCo and others, and to test the CVU sign flip.
2. **Make the PAC-Bayes link quantitative:** compute the actual Catoni / Gibbs bound with the init prior, and compare `F(λ)` with an optimized λ against the true generalization gap, not just test accuracy.
3. ~~Optimise temperature per architecture~~ (done in v2: no gain over fixed normalization; the 0-1 loss removes the scale problem instead).
6. **Estimate the lower tail better than best-of-K:** importance sampling / annealing from $P_0$ toward low training error, or the NNGP estimate of Valle-Pérez et al., and compare the two head to head.
4. **Is there information beyond CVU?** Try label-permutation controls (variance under random labels vs. true labels) and the third cumulant.
5. Use the remaining loss-variance terms (`var_data`) as a *second* feature next to CVU/ZiCo in a small rank-ensemble.

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_gradcheck.py
python experiments/run_sweep.py --n-archs 150 --out results/sweep_full.jsonl                     # ~6 min on 2 CPUs
python experiments/run_sweep.py --n-archs 150 --n-train 200 --epochs 150 --out results/sweep_n200.jsonl
python experiments/run_gain_sweep.py --out results/gain_sweep.jsonl                             # ~25 min
python experiments/recompute_proxies.py results/sweep_n200.jsonl --n-train 200 --probe 200       # if proxies change
python experiments/analyze.py results/*.jsonl
python experiments/make_figures.py
# v2 (full grids, K=128, 0-1 loss proxies, Fashion-MNIST); ~45 min on 72 CPUs
bash experiments/run_v2.sh digits && bash experiments/run_v2.sh fashion
python experiments/summarize_v2.py > results/v2/summary.md
```

## References

- Gracheva, E. (2021). Trainless model performance estimation based on random weights initialisations for neural architecture search. *Array* 12. arXiv:2103.08312
- Gracheva, E. (2023/2025). Neural architecture search: two constant shared weights initialisations (epsinas). arXiv:2302.04406
- Abdelfattah, M. et al. (2021). Zero-cost proxies for lightweight NAS. ICLR. arXiv:2101.08134
- Mellor, J. et al. (2021). Neural architecture search without training. ICML.
- Li, G. et al. (2023). ZiCo: zero-shot NAS via inverse coefficient of variation on gradients. ICLR. arXiv:2301.11300
- Týbl, O., Neumann, L. (2025). Training-free NAS through variance of knowledge of deep network weights. CVPR.
- Ru, B., Lyle, C. et al. (2021). Speedy performance estimation for NAS (SoTL). NeurIPS. arXiv:2006.04492
- Lotfi, S. et al. (2022). Bayesian model selection, the marginal likelihood, and generalization. ICML. arXiv:2202.11678
- Bowden, J., Csaposs, S., Hoffmann, T. (2021). PAC-Bayes inspired NAS without training. Caltech CS159 report.
- Valle-Pérez, G., Camargo, C. Q., Louis, A. A. (2019). Deep learning generalizes because the parameter-function map is biased towards simple functions. ICLR. arXiv:1805.08522
- Valle-Pérez, G., Louis, A. A. (2020). Generalization bounds for deep learning. arXiv:2012.04115
- Mingard, C., Valle-Pérez, G., Skalse, J., Louis, A. A. (2021). Is SGD a Bayesian sampler? Well, almost. JMLR 22. arXiv:2006.15191
- Germain, P., Bach, F., Lacoste, A., Lacoste-Julien, S. (2016). PAC-Bayesian theory meets Bayesian inference. NeurIPS. arXiv:1605.08636
- Park, D. S., Lee, J., Peng, D., Cao, Y., Sohl-Dickstein, J. (2020). Towards NNGP-guided neural architecture search. arXiv:2011.06006
- Picard-Weibel, A., Clerico, E., Moscoviz, R., Guedj, B. (2025). How good is PAC-Bayes at explaining generalisation? arXiv:2503.08231
