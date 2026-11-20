# Experiments

All runs on kipchoge (Xeon w9-3475X, 72 cores; CPU only). NumPy experiments use the repo `.venv`
(`PYTHONPATH=. OMP_NUM_THREADS=1`); PyTorch experiments use `~/.venv` (torch 2.5.1). Fashion-MNIST comes from the
Zalando IDX files (`lvproxy/data.py`), cached in `~/scikit_learn_data/fashion`.
Status: **final** (numbers stand), **superseded** (kept for the record; see replacement), **preliminary**.

| ID | Question | Script | Results | Status |
|---|---|---|---|---|
| E1–E3 | Does init-loss variance predict trained accuracy? (v1) | `run_sweep.py`, `run_gain_sweep.py` | `results/*.jsonl`, `results/correlation_tables.md` | final (superseded in scope by v2) |
| E-v2 | Same, with full grids, K = 128, 0-1 loss proxies, Fashion-MNIST | `run_v2.sh`, `summarize_v2.py` | `results/v2/{grid_full,grid_n200,gain_sweep,fashion_full,fashion_n1000}.jsonl`, `summary.md`, `tables_*.md` | final |
| E-v2-tail | Is the lower tail determined by the variance? | `tail_vs_variance.py` | printed | final |
| T-moment | Worst-case Catoni bound given prior mean/variance | `moment_worst_case.py` | printed (theory.md §4) | final |
| E-fkl | Parameter- vs function-space KL of PAC-Bayes posteriors | `function_space_kl.py`, `summarize_fkl.py` | `results/v2/function_space_kl.jsonl` | final (no MC correction; see note) |
| E-norm | Normalization in PAC-Bayes; invariance-aware KL | `pacbayes_norm.py`, `summarize_norm.py` | `results/v2/pacbayes_norm.jsonl`, `norm_summary.md` (block estimator); `*_m150.jsonl` superseded | final |
| E-normdet | Ordinary training: none / LN / BN | `pacbayes_norm.py --deterministic` | `results/v2/norm_deterministic.jsonl` | final |
| E-r1 | Pull-to-init sweep, hybrid fine-tuning, 3 seeds; variational seeds | `pacbayes_sgd.py --grid custom --finetune`, `pacbayes_norm.py --seeds`, `summarize_seeds.py` | `results/v2/sgd_r1_wdsweep.jsonl`, `sgd_r1_summary.md`, `pacbayes_norm_seeds.jsonl`, `norm_seeds_summary.md` | final |
| E-r2 | f_quad objective (variational, hybrid); BN × SGD temperature × pull | same | `results/v2/r2a_var_quad.jsonl`, `r2b_*.jsonl`, `r2a_summary.md`, `r2b_summary.md` | final |
| E-r3 | Exact kl⁻¹ objective vs the García-Pérez accuracy objective (klinv01); hybrid with klinv01 | same, `--objective klinv / klinv01` | `results/v2/r3_*.jsonl` | final (`objectives_summary.md`) |
| E-online | Trajectory certificates from one SGD pass: progressive validation and online PAC-Bayes (OPBTrain) | `online_bounds.py`, `summarize_online.py` | `results/v2/online_bounds.jsonl`, `online_summary.md` | final |
| E-dp | Certificate for a DP-SGD-trained network via mutual information (+ Markov); Fashion grid and pre-registered MNIST transfer; Gaussian PAC-Bayes on MNIST | `dp_sgd_bound.py`, `summarize_dp.py`, `run_mnist_transfer.sh`, `pacbayes_norm.py` | `results/v2/dp_sgd_r{1,2,3_*}.jsonl`, `dp_transfer_*.jsonl`, `mnist_pb_*.jsonl`, `dp_summary.md` | final |
| E-sgd | SGD-noise subspace posterior; lr × batch | `pacbayes_sgd.py`, `summarize_sgd.py` | `results/v2/pacbayes_sgd.jsonl`, `sgd_summary.md`; `pacbayes_sgd_m150.jsonl` superseded | final |

---

## E-v2: init-variance proxies at scale

**Setup.** Depth {1,2,4,8,12} × width {16,32,64,128} × activation {relu,tanh,sigmoid,linear,gelu,sin} × residual × gain
{0.5,1,1.414,2,3} = 1,200 MLPs; proxies from K = 128 inits on a probe batch (256; 200 for n = 200); 3 training seeds.
Datasets: digits full (40 epochs), digits n = 200 (150 epochs), gain sweep (depth 8, width 64, 120 configs), Fashion
10k / 5k test (20 epochs), Fashion 1k train (100 epochs). Command: `bash experiments/run_v2.sh {digits|fashion}`.

**Results** (Spearman ρ with test accuracy; parentheses: partial ρ given CVU):

| proxy | digits | digits n=200 | gain sweep | Fashion 10k | Fashion 1k |
|---|---|---|---|---|---|
| CE var (normalized) | +0.26 (−0.10) | +0.33 (−0.10) | +0.32 (−0.25) | +0.04 (−0.23) | +0.21 (−0.16) |
| 0-1 var (acc_var_init) | +0.32 (−0.17) | +0.41 (−0.16) | +0.50 (+0.08) | +0.22 (+0.16) | +0.36 (+0.16) |
| best-of-128 untrained acc | +0.32 (+0.05) | +0.39 (+0.06) | +0.57 (+0.34) | +0.24 (+0.15) | +0.36 (+0.16) |
| permuted-label 0-1 var | −0.19 | −0.00 | +0.12 | +0.16 | +0.09 |
| CVU | +0.37 | +0.46 | +0.49 | +0.19 | +0.34 |
| log #params | +0.16 | +0.07 | — | +0.19 | +0.23 |

Findings: 0-1 variance > CE variance everywhere; the lower tail beats the variance; signal mostly label-dependent;
|ρ| modest (≈ log #params on Fashion); among architectures with acc > 0.5, ρ ≈ 0.12–0.37.
**Conclusion recorded 2026-11-19: the init-variance proxy is not a promising direction on its own** (theory.md §6
explains why: it measures the segment of the free-energy curve that localization removes).
Open loose end: on digits, mean untrained accuracy correlates −0.28 with trained accuracy although every architecture's
mean is within sampling error of 0.1.

## E-v2-tail

ρ(best-of-128, std of untrained acc) = 0.80–0.81 in all five experiments; (best − mean)/std median ≈ 3.0 (IQR 2.7–3.5);
that shape term predicts accuracy with ρ +0.12 to +0.41; best | std partial +0.10 to +0.33; std | best +0.04 to +0.18.

## E-fkl: parameter- vs function-space KL

**Setup.** Fashion 60k train / 10k test. Prior N(m₀, σ₀²), m₀ ~ N(0, 1/fan_in) random; diagonal Gaussian posterior,
50 epochs Adam (lr 5e-3) on E_Q[bounded CE] + √((KL + ln(2√n/δ))/2n). Depth {1,2} × width {32,64,128} ×
{relu, tanh} × σ₀ {0.02, 0.05}. KL_m on logits (softmax shift removed) at m training inputs:
(i) shared-Jacobian linear model (exact push-forward, respects data processing); (ii) MC Gaussian fit (no
linearization). Digits (n = 1,347) was abandoned: the bound-optimal posterior there is the prior itself.

**Results.** Share of KL_param visible to the linear model (32-wide nets, where m reaches 160):

| | m = 10 | m = 40 | m = 160 |
|---|---|---|---|
| σ₀ = 0.02 | 28–46% | 54–70% | 75–85% (rising) |
| σ₀ = 0.05 | 19–29% | 39–48% | 58–63% (rising) |

92–99% of KL_param is the mean term. MC Gaussian fit unreliable (exceeds KL_param in one run: 3,551 > 2,537).
Conclusion: at small σ, re-measuring a Gaussian KL in function space saves little (upper bound on saving ≈ 15–25% at
σ₀ = 0.02). Note: certificates here (0.40–0.55) were computed without the MC sampling correction.

## E-normdet: ordinary training

Adam 1e-3, 20 epochs, depth 2, Fashion n = 55k, 3 seeds. Test error width 64 / 128: none 0.1132 / 0.1114,
LN 0.1148 / 0.1108, **BN 0.1113 / 0.1076**, reference-set BN 0.1145 / 0.1080. BN best and most stable; LN no gain.

## E-norm: normalization in PAC-Bayes

**Setup.** Fashion: S = 55k labelled, R = 1,024 unlabelled reference inputs held out from S (only for reference-set
BN), 10k test. Depth 2, width {64,128}, σ₀ {0.02, 0.05}, 50 epochs Adam 5e-3 on the McAllester-type objective.
Variants: none, ln, bn_ref, bn_param (naive statistics as parameters), bn_stat (two-phase statistics as parameters);
mode plain / inv (invariance log-scales trained) + post-hoc LBFGS over the invariance group. PAC-Bayes-kl, δ = 0.025,
MC correction δ′ = 0.01.

**Superseded results (M = 150 full samples; correction ≈ 0.035 nats/example inflates certificates by up to ~0.11).**
Relative comparisons held: inv training cut KL by 30–65% for every variant; post-hoc rescaling saved <1%;
bn_stat + inv gave the lowest KL in every setting (775–1,112 nats) and the best certificate at σ₀ = 0.02;
naive bn_param was the worst BN variant (statistics cost 360–780 nats).

**Block-estimator results (final; `results/v2/norm_summary.md`).** Certificate (KL in nats), plain → inv:

| σ₀ | width | none | LayerNorm | BN, reference set | BN stats naive | **BN stats as params** |
|---|---|---|---|---|---|---|
| 0.02 | 64 | 0.507 → 0.445 (2294 → 1540) | 0.511 → 0.448 (2079 → 1097) | 0.499 → 0.451 (2238 → 1038) | 0.499 (2611) | 0.477 → **0.407** (1761 → 775) |
| 0.02 | 128 | 0.498 → 0.489 (2835 → 1922) | 0.505 → 0.438 (2729 → 1242) | 0.525 → 0.449 (3227 → 1136) | 0.507 (4162) | 0.476 → **0.394** (2378 → 949) |
| 0.05 | 64 | 0.429 → 0.399 (1753 → 1162) | 0.448 → **0.397** (1528 → 1018) | 0.438 → 0.405 (1663 → 1060) | 0.465 (2335) | 0.420 → **0.395** (1402 → 803) |
| 0.05 | 128 | 0.521 → 0.446 (3192 → 2024) | 0.482 → **0.414** (2257 → 1153) | 0.465 → 0.418 (2580 → 1169) | 0.497 (3513) | 0.455 → **0.414** (2386 → 1112) |

- The block estimator lowers every certificate by ≈0.10–0.12 relative to M = 150 (same posteriors).
- BN statistics as parameters + invariance-aware training: best certificate in 3 of 4 settings, tied in the 4th
  (0.394–0.414); lowest KL everywhere. Naive statistics-as-parameters is the worst BN variant.
- Invariance-aware training lowers the certificate by 0.01–0.08 for every variant; post-hoc rescaling changes it by
  <0.001.
- Differences among the best variants at σ₀ = 0.05 are ≈0.01–0.02; single seed, so not yet significant.
- Compare E-sgd: SGD with L2 pull to init + prior translated to the SGD mean reaches 0.374 (none, width 64) with a much
  larger KL (≈4,400) but much lower Gibbs error (0.18 vs 0.29–0.31): direct bound optimization with Adam finds a
  worse error/KL trade-off.

## E-sgd: SGD-noise subspace posterior

**Setup.** As E-norm data. SGD momentum 0.9, constant lr, 30 epochs from m₀ (bn_stat: batch statistics for 15 epochs,
then statistics frozen into parameters and SGD continues in inference mode); T = 60 iterates over the last 10 epochs;
optional L2 pull to m₀ (wd_init). Posterior per theory.md §11; σ₀ ∈ {0.01, 0.02, 0.05, 0.1, 0.2} with union bound;
K ∈ {0, 10, 40}, α ∈ {0.25 … 64}, τ ∈ {0.05 … 1} selected on a 5k training subset with 4 samples; final certificate
with the block estimator (b = 40, 20 passes). Grids: {none, bn_stat} × width {64,128} × wd_init {0, 1e-3, 1e-2}
(lr 0.05, batch 128); and lr {0.01, 0.05, 0.2} × batch {32, 128, 512} at width 128, wd_init 1e-3.

**Preliminary (superseded, M = 150):** best certificate 0.497 (none, width 64, wd_init 1e-2; selection estimate 0.358);
wd_init 1e-2 best (‖θ̄ − m₀‖ ≈ 4.7 vs 13–40); plain SGD diverges at lr 0.2 with batch 32/128 and at lr 0.05 with batch 32
(test error 0.65–0.90) while bn_stat trains (0.11–0.16).

**Block-estimator results (final; `results/v2/sgd_summary.md`).** The grid has 28 configurations.

| norm | width | L2-to-init | ‖θ̄−m₀‖ | SWA test | Gibbs test | cert (best) | cert (isotropic K=0) | chosen | KL (mean term) |
|---|---|---|---|---|---|---|---|---|---|
| none | 64 | 0 | 35.1 | 0.104 | 0.179 | 0.679 | 0.677 | K=10, τ=0.5, σ₀=0.2 | 32,966 (15,385) |
| none | 64 | 1e-3 | 13.4 | 0.103 | 0.180 | 0.520 | 0.517 | K=10, τ=0.7, σ₀=0.1 | 14,743 (9,033) |
| none | 64 | **1e-2** | 4.7 | 0.128 | 0.184 | **0.374** | **0.369** | K=10, τ=1, σ₀=0.05 | 4,434 (4,333) |
| none | 128 | 1e-2 | 4.7 | 0.123 | 0.205 | 0.400 | 0.389 | K=40, τ=1, σ₀=0.05 | 4,578 (4,383) |
| bn_stat | 64 | 1e-2 | 4.6 | 0.128 | 0.301 | 0.502 | 0.477 | K=40, τ=1, σ₀=0.05 | 4,441 (4,233) |
| bn_stat | 128 | 1e-2 | 4.7 | 0.124 | 0.313 | 0.516 | 0.517 | K=0, τ=1, σ₀=0.05 | 4,344 (4,344) |

Findings:
- **The SGD subspace adds nothing**: the isotropic posterior certifies the same or better everywhere; subspace KL terms
  are 50–800 nats vs a mean term of ~4,300. The winning posterior has τ = 1, i.e. it is *the prior translated to the
  SGD average*. The Lyapunov-subspace hypothesis (theory.md §11) is not supported here.
- **The L2 pull to m₀ is the lever**: wd_init 1e-2 keeps the solution 4.7 from m₀ (13–40 otherwise) and halves the
  certificate relative to no pull. Larger pulls not yet tried.
- **bn_stat was hurt by a design flaw** (fixed): in whitened coordinates the BN statistics got the same relative width
  τ, i.e. noise of e^{±0.5} on every unit's scale at τ = 1. **Fix** (`pacbayes_sgd_bnfix.jsonl`): the statistics form an
  independent block N(z̄_s, τ_s² I) with their own grid τ_s ∈ {0.02, 0.1, 0.3, 1}. Result at wd_init 1e-2: width 64
  0.502 → **0.386**, width 128 0.516 → **0.413** (no normalization: 0.374 / 0.400); τ_s ≈ 0.1 is selected; Gibbs test
  error 0.31 → 0.19–0.20.
- **SGD temperature and certifiability (bn_stat, width 128, wd_init 1e-3, 9 runs):** Spearman(lr/B, cert) = −0.90:
  certificates improve from 0.70–0.73 at lr/B ≈ 1e-4 to 0.52 at lr/B ≈ 1.6e-3, with larger σ₀ (0.1) and τ = 1 selected.
  But Spearman(lr/B, SWA test error) = +0.48 and Spearman(cert, SWA test) = −0.50: the certificate tracks the noise
  level of SGD (robustness to prior-sized perturbations), not test error. Plain networks: no relation (ρ = −0.03;
  high temperatures diverge). Single seed.
- **lr × batch** (width 128, wd 1e-3): SWA test ≈ 0.10 for all stable runs; certificates 0.58–0.80 without a clear
  relation to test error (posterior selection noise dominates). No support for D.4 yet.
- **BN enables high learning rates**: plain SGD diverges at lr 0.2 with batch 32 (test 0.90) and 128 (0.83) and at
  lr 0.05 with batch 32 (0.65); bn_stat trains (0.16, 0.12, 0.11).
- Block estimator vs M = 150 on the same configuration (none, w64, wd 1e-2): 0.374 vs 0.497.

## E-r1: pull-to-init sweep, hybrid, seeds (3 seeds; mean ± sd)

SGD (lr 0.05, batch 128, 30 epochs) with L2 pull to m₀ ∈ {1e-2, 3e-2, 1e-1}; certificates for the SGD posterior, the
isotropic posterior (prior translated to the SGD mean), and the hybrid (SGD warm start + 10 epochs of direct
optimization of the McAllester-type surrogate). Full table: `results/v2/sgd_r1_summary.md`.

| norm | width | pull | ‖θ̄−m₀‖ | SWA test | cert isotropic | cert hybrid | hybrid Gibbs test / KL |
|---|---|---|---|---|---|---|---|
| none | 64 | 1e-2 | 4.64 | 0.127 | 0.370 ± 0.001 | 0.389 ± 0.004 | 0.293 / 816 |
| none | 64 | **3e-2** | 3.07 | 0.148 | **0.355 ± 0.002** | 0.387 ± 0.004 | 0.291 / 822 |
| none | 64 | 1e-1 | 2.11 | 0.177 | 0.382 ± 0.002 | 0.388 ± 0.003 | 0.293 / 825 |
| none | 128 | 3e-2 | 3.06 | 0.144 | 0.379 ± 0.002 | 0.410 ± 0.001 | 0.305 / 976 |
| bn_stat | 64 | 3e-2 | 3.04 | 0.148 | 0.376 ± 0.001 | 0.385 ± 0.003 | 0.294 / 758 |
| bn_stat | 128 | 3e-2 | 3.04 | 0.144 | 0.404 ± 0.003 | 0.395 ± 0.007 | 0.295 / 960 |
| bn_stat | 128 | 1e-1 | 2.05 | 0.172 | 0.418 ± 0.002 | 0.392 ± 0.003 | 0.299 / 762 |

Variational (inv mode, McAllester surrogate), 3 seeds: bn_stat 0.395–0.411 (sd ≤ 0.008); LN 0.399–0.453;
none 0.417–0.491 (sd up to 0.023). `results/v2/norm_seeds_summary.md`.

Findings:
- **Best certificate: 0.355 ± 0.002** (no normalization, width 64, pull 3e-2, prior translated to the SGD mean).
  Optimum pull ≈ 3e-2; at 1e-1 the SGD solution's error (0.177) dominates.
- **Seed noise is small** (sd 0.001–0.007 for SGD posteriors), so differences ≥ 0.01 are meaningful.
- **The hybrid converges to the variational regime** (Gibbs test ≈ 0.29–0.30, KL ≈ 760–1,000) from any warm start,
  and is worse than the SGD posteriors except where those are weak. Diagnosis: the McAllester surrogate
  (√((KL + c)/2n)) over-penalizes KL relative to the kl-inversion certificate in the low-error regime. → round 2 tests
  the f_quad objective (refined Pinsker; Pérez-Ortiz et al.).
- **bn_stat is the best and most stable variational method**; the translated-prior SGD posterior beats the best
  variational posterior by ≈0.04.

## E-r2: f_quad objective; BN × SGD temperature × pull (3 seeds)

**Variational, inv mode, f_quad vs McAllester** (`r2a_summary.md` vs `norm_seeds_summary.md`):

| variant | width | σ₀ | McAllester | f_quad |
|---|---|---|---|---|
| bn_stat | 64 | 0.02 | 0.397 ± 0.008 | 0.383 ± 0.008 |
| bn_stat | 64 | 0.05 | 0.395 ± 0.003 | **0.379 ± 0.002** |
| bn_stat | 128 | 0.02 | 0.398 ± 0.003 | 0.387 ± 0.004 |
| bn_stat | 128 | 0.05 | 0.411 ± 0.002 | 0.396 ± 0.003 |
| ln | 64 | 0.05 | 0.399 ± 0.002 | 0.387 ± 0.002 |
| none | 64 | 0.05 | 0.417 ± 0.023 | 0.404 ± 0.017 |

f_quad improves every variational setting by 0.011–0.016 (Gibbs test error ≈ 0.28 vs 0.30), confirming that the
surrogate objective, not the posterior family, limited variational training.

**SGD + hybrid (f_quad fine-tuning)** (`r2b_summary.md`):
- bn_stat hybrid: **0.365 ± 0.001** (width 64, lr 0.05, batch 32, pull 3e-2) and 0.368–0.369 at batch 128, beating
  both the bn_stat SGD posterior (0.376) and pure variational bn_stat (0.379). With the McAllester surrogate the hybrid
  was worse than the SGD posterior (round 1).
- none: the translated prior is still best (0.354 ± 0.001 isotropic) vs hybrid 0.373.
- **Temperature × pull:** with pull ≥ 1e-2, raising lr to 0.2 worsens the SGD solution (test 0.17–0.20) and the
  certificate (0.41–0.45); lr 0.2 with batch 32 diverges in 7 of 12 bn_stat runs (the hybrid still rescues the
  0.90-error solutions to ≈0.40). The round-0 correlation "higher lr/B → better certificate" (wd 1e-3, σ₀ = 0.1
  selected) does not carry over to the strong-pull regime.

## E-r3: exact kl⁻¹ objective vs the García-Pérez accuracy objective (3 seeds)

`klinv`: exact kl⁻¹ of the bound on the bounded CE (implicit-function gradients). `klinv01`: García-Pérez et al.
(2025, eq. 9), kl⁻¹ at the rolling 0-1 risk with gradient m·∇L_xe (straight-through). Status at 19:16, 2026-11-19:
all 60 runs complete (width-128 variational collected 20:42). Full comparison of the four objectives:
`results/v2/objectives_summary.md`.

**Variational, inv mode, width 64:**

| variant | σ₀ | McAllester | f_quad | klinv (CE) | **klinv01 (accuracy)** | klinv01 Gibbs test / KL |
|---|---|---|---|---|---|---|
| bn_stat | 0.02 | 0.397 | 0.383 | 0.372 ± 0.007 | **0.358 ± 0.001** | 0.198 / 2,969 |
| bn_stat | 0.05 | 0.395 | 0.379 | 0.364 ± 0.009 | **0.348 ± 0.001** | 0.191 / 3,002 |
| none | 0.02 | 0.472 | 0.470 | 0.446 ± 0.008 | 0.462 ± 0.004 | 0.218 / 6,549 |
| none | 0.05 | 0.417 | 0.404 | 0.394 ± 0.017 | 0.398 ± 0.009 | 0.203 / 4,418 |

**Hybrid (SGD lr 0.05, batch 128, pull 3e-2, then 10 epochs of klinv01):**

| norm | width | SGD posterior | translated prior (isotropic) | **hybrid klinv01** | Gibbs test / KL |
|---|---|---|---|---|---|
| bn_stat | 64 | 0.376 | 0.375 | **0.347 ± 0.001** | 0.192 / 2,930 |
| bn_stat | 128 | 0.406 | 0.405 | 0.364 ± 0.002 | 0.193 / 3,427 |
| none | 64 | 0.356 | 0.354 | **0.348 ± 0.002** | 0.191 / 3,017 |
| none | 128 | 0.384 | 0.379 | 0.373 ± 0.004 | 0.196 / 3,703 |

Findings:
- **New best certificates: 0.347 ± 0.001** (bn_stat hybrid) and **0.348 ± 0.001** (bn_stat variational alone,
  σ₀ = 0.05; and plain-network hybrid). Previous best: 0.354 (translated prior).
- **Objective ladder for variational bn_stat (width 64, σ₀ = 0.05): McAllester 0.395 → f_quad 0.379 → exact kl⁻¹ on CE
  0.364 → accuracy objective 0.348.** Each step aligns the training objective more closely with the 0-1 certificate.
  The accuracy objective moves to the low-error / high-KL regime (Gibbs test 0.19, KL ≈ 3,000), the regime where the
  SGD posteriors certified well.
- **With the accuracy objective, BN statistics as parameters is needed**: plain networks do not benefit (0.398 vs
  0.394; KL up to 6,500, larger seed variance), while bn_stat gains 0.016 over the exact kl⁻¹ and has tiny seed variance.
- With the accuracy objective, pure variational bn_stat matches the SGD hybrid (0.348 vs 0.347): the SGD warm start is
  no longer needed for BN networks.

### Objective comparison, all settings (variational, inv mode, 3 seeds)

| model | width | σ₀ | McAllester | f_quad | kl⁻¹, CE target (eq. 8) | kl⁻¹, 0-1 target (eq. 9) |
|---|---|---|---|---|---|---|
| bn_stat | 64 | 0.02 | 0.397 | 0.383 | 0.372 | **0.358** |
| bn_stat | 64 | 0.05 | 0.395 | 0.379 | 0.364 | **0.348** |
| bn_stat | 128 | 0.02 | 0.398 | 0.387 | 0.375 | **0.366** |
| bn_stat | 128 | 0.05 | 0.411 | 0.396 | 0.372 | **0.366** |
| none | 64 | 0.02 | 0.472 | 0.470 | **0.446** | 0.462 |
| none | 64 | 0.05 | 0.417 | 0.404 | **0.394** | 0.398 |
| none | 128 | 0.02 | 0.491 | 0.490 | **0.474** | 0.492 |
| none | 128 | 0.05 | 0.451 | 0.450 | 0.452 | 0.465 |

- Exact kl⁻¹ (implicit gradients) beats f_quad in 7 of 8 settings and ties the 8th: relaxations only distort the
  gradient direction (García-Pérez et al. Sec. 4.1). **Decision (2026-11-19): drop f_quad / McAllester as training
  objectives; keep them as baselines.**
- The 0-1 target (eq. 9) wins for every bn_stat setting but loses for every plain-network setting (KL 6,500–8,000 vs
  2,700–3,200 with the CE target). Default: eq. 8; use eq. 9 where it helps (BN statistics as parameters).

## E-online: trajectory-based certificates from one SGD pass (3 seeds)

One pass over S (55k, fixed random order), SGD momentum 0.9, {none, LN} × width {64, 128} × lr {0.01, 0.05} × batch
{32, 128}. (a) Progressive validation (= OPBTest of Haddouche & Guedj 2022): predictor "random iterate scoring an
example in the last fraction f of the pass", f ∈ {1, 0.5, 0.25, 0.1} with union bound, Chernoff-kl for martingale
differences, δ = 0.025. (b) OPBTrain with Gaussian noise σ ∈ {0.003 … 0.3} around the iterates (step energy
b·Σ‖Δ_j‖²/2σ²), union bound over σ × λ (13 values). Theory: theory.md §13. Table: `results/v2/online_summary.md`.

- **Progressive validation: 0.148–0.159** for all stable configurations (sd ≤ 0.005); measured test error of the
  certified iterates 0.138–0.148 (gap ≈ 0.01). Best 0.148 (LN, width 128, lr 0.01, batch 32). 2.3× tighter than the best
  data-free PAC-Bayes certificate (0.347), but it certifies a single-pass, random-iterate learner and is a test bound
  in disguise (each example is scored before it is trained on). Unstable runs (none, lr 0.05, batch 32: one-pass test
  0.32–0.59) are certified accordingly (0.28–0.37).
- **OPBTrain: vacuous everywhere (≥ 0.87).** Best configuration: σ = 0.003 → 621 nats/example of step energy;
  σ = 0.03 → 6.2 nats/example (noisy error 0.17); σ = 0.1 → 0.56 nats/example but noisy error 0.43; σ = 0.3 → error
  0.87. SGD's steps (‖Δ‖ ≈ 0.1 with momentum) are large and poorly aligned, so the per-step KL dominates: **SGD's noise
  makes the trajectory-KL route expensive, not cheap.**

## E-dp: certifying a network trained with noisy SGD (mutual information, theory.md §14)

DP-SGD (per-example clip C = 1, Poisson sampling q = B/n, Gaussian noise ν·B·C on the gradient sum, SGD momentum 0.9);
certificate kl(train‖risk) ≤ ((I_ub + 1/e)/δ₁ + ln(2√n/δ₂))/n with δ₁ = 0.024, δ₂ = 0.001, I_ub = n·T·KL_step(q, σ)
computed by quadrature. Data: Fashion (55k, our split) and MNIST (standard 60k/10k). Gabor features: data-free
first-order scattering-like map (2 scales × 8 orientations, modulus, 4×4 pooling, per-example standardization) on raw
[0,1] pixels. Tables: `results/v2/dp_summary.md`.

Rounds on Fashion (selection, ~120 runs):
1. LN/plain MLP 784-128-128-10 on pixels, B ∈ {1024, 4096}, ν ∈ {0.005, 0.01, 0.02}, 10–30 epochs, lr ∈ {0.5, 2}:
   best 0.335 (B 4096, ν 0.005, 10 epochs; train 0.211, test 0.207, I = 49). Gap ≈ 0; more epochs only add I.
2. + cosine decay and tail averaging, B up to 16384: best 0.330 (I = 24). **At fixed (T, ν) the batch size is
   irrelevant** (B 4096 × 5 epochs ≡ B 16384 × 20 epochs to 4 digits) and I = T/(2ν²n) exactly; averaging does not help.
3. Linear softmax on Gabor features / on pixels, and 1-hidden-layer heads: **best 0.2745** (linear on Gabor, ν 0.01,
   T = 134, I = 12.2); linear on (standardized) pixels 0.29; MLP on Gabor 0.276.

Pre-registered transfer (recipes A and B fixed from round 3, applied unchanged; 3 seeds):
- **MNIST recipe A: 0.099 ± 0.004** (train 0.062, test 0.058, I = 12.2). Recipe B: 0.116 ± 0.003.
- Fashion recipe A: 0.273 ± 0.003 (seed 0 from the selection grid).
- **CIFAR-10 recipe A: 0.740 ± 0.005** (test 0.675; Gabor map per RGB channel), recipe B 0.761 ± 0.006. Ties the best
  published data-free certificate (García-Pérez et al. 2025, 4-layer CNN, 0.738 ± 0.007); Lampert & Zakerinia 0.84.
- MNIST, Gaussian PAC-Bayes pipeline (width 64, σ₀ 0.05, block estimator): bn_stat + inv + klinv01 0.228 ± 0.002
  (Gibbs test 0.067, KL 5,400); plain + inv + klinv01 0.267 ± 0.004; Pérez-Ortiz-style plain + f_quad 0.349 ± 0.021.

Findings:
- **The MI route beats every data-free Gaussian PAC-Bayes certificate we have** (Fashion 0.273 vs 0.347) and, on MNIST,
  the published data-free certificates we know (Lotfi et al. 0.116, Pérez-Ortiz CNN 0.2165, Lampert & Zakerinia 0.29),
  for a *deterministic* network rather than a Gibbs predictor. The model is weak (5.8% MNIST test error). On MNIST,
  our Gaussian pipeline (0.228) also beats García-Pérez et al.'s best MLP certificate (0.315, 3×600 MLP).
- The regime is extreme privacy: I ≈ 12 nats for 60k examples. The certificate is limited by (i) the Markov factor
  1/δ₁ = 42 (no-Markov value 0.069 on MNIST) and (ii) training error under heavy noise (noise norm ∝ √d, hence small
  models and good data-free features win).
- Gaussian PAC-Bayes on MNIST confirms the Fashion ordering: BN statistics as parameters > plain (0.228 vs 0.267), and
  both ≫ the f_quad / plain-KL baseline (0.349).
- Caveats: Fashion pixels are standardized with training-set statistics (all earlier runs too; MNIST uses fixed
  constants, Gabor runs use raw pixels). Fashion recipe numbers carry unpaid selection.
