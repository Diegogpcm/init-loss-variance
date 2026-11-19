# Theory notes

Derivations and arguments developed in this project, with assumptions stated. Notation: P₀ is the prior (the
initialization distribution unless stated otherwise), R̂(θ) the empirical 0-1 risk on n training examples, L(θ) the
population risk, C the number of classes, Q a posterior. Status tags: **[standard]** known result restated,
**[ours]** derived here (novelty not established unless said so), **[numerical]** checked only numerically.

---

## 1. PAC-Bayes optimum as a free energy [standard]

For Catoni's bound (Picard-Weibel convention, λ multiplies the KL),
B(Q) = Q[R̂] + λ·KL(Q‖P₀) + λ·ln(1/δ) + 1/(8nλ),
the minimum over Q is reached by the Gibbs posterior dQ/dP₀ ∝ e^{−R̂/λ}, and equals
min_Q B = −λ ln E_{P₀} e^{−R̂/λ} + λ ln(1/δ) + 1/(8nλ)   (Donsker–Varadhan).
The first term is a free energy: a functional of the *push-forward of P₀ by R̂* (the "risk prior"). Picard-Weibel
et al. (2025, Thm 1) prove the same for every bound of the form b(Q[R̂], P₀[D(dQ/dP₀)]).

**Cumulant expansion.** With μ₀, σ₀² the mean and variance of R̂ under P₀ and β = 1/λ:
−(1/β) ln E e^{−βR̂} = μ₀ − (β/2)σ₀² + O(β²κ₃). At a fixed mean, more variance lowers the free energy.
This was the original motivation of the project (prior variance of the init loss as an architecture score).
**Limitation:** with β ~ n the expansion is meaningless; the free energy is governed by the lower tail of the risk
prior (§3, §4).

## 2. Symmetry pins the prior mean of the 0-1 risk [ours, elementary]

If the output layer has i.i.d. weights and zero bias, the logits are exchangeable across classes, so for every input
P(argmax = y) = 1/C exactly. Hence E_{P₀}[R̂] = E_{P₀}[L] = 1 − 1/C, independently of the data distribution.
Consequences:
- every architecture comparison is automatically at equal mean (the regime of §3);
- any certificate must beat 1 − 1/C to be informative;
- the CE loss has no such property (its mean depends on logit scale), which is why CE variance needed temperature
  normalization and the 0-1 variance did not.
Empirically: the mean untrained accuracy is 0.100 within sampling error for ~99% of the 4,920 architectures
(experiments.md, E-v2-tail).

## 3. Variance vs. tail: convex order [ours, short proof]

**Variance alone does not order bounds.** Example (n = 1000, δ = 0.05, all means 0.5), best Catoni bound over Q and λ:

| risk prior | variance | min risk | best bound |
|---|---|---|---|
| A: 0.3 / 0.7 (½, ½) | 0.040 | 0.30 | 0.343 |
| B: 0.4 (0.82) / 0.95 (0.18) | 0.045 | 0.40 | 0.440 |
| C: 0.1 / 0.5 / 0.9 (¼, ½, ¼) | 0.080 | 0.10 | 0.147 |

B has more variance than A but a worse bound: its extra variance sits in the upper tail.

**Convex order does.** If ρ₂ is a mean-preserving spread of ρ₁ (ρ₁ ≤_cx ρ₂), then since r ↦ e^{−r/λ} is convex,
E_{ρ₂} e^{−R/λ} ≥ E_{ρ₁} e^{−R/λ} for every λ, so the optimized Catoni bound of ρ₂ is no worse. This is the correct
formal version of "more spread at fixed mean is better". Picard-Weibel et al. use only first-order stochastic dominance,
which cannot compare distributions with equal means (they note all such distributions are incomparable), so this case
is not covered by their results. *Checked for Catoni's bound only; extension to their class B(D, b) is open.*

**Empirical link (0-1 loss).** Because the mean is pinned near the boundary (§2), extra variance must come mostly from
the low-risk tail: ρ(best-of-128 untrained acc, std) ≈ 0.80 in all experiments, and the best init sits a median 3.0
SDs above the mean (Gaussian expectation for K = 128 ≈ 2.6, so the high-accuracy tail is heavier). The residual shape
(best − mean)/std still predicts trained accuracy (ρ +0.12 to +0.41).

## 4. What the prior's mean and variance can certify: moment worst case [ours, numerical]

Given only μ = 1 − 1/C and σ² of the prior 0-1 risk, the guaranteed Catoni bound is the worst case over all risk
distributions on [0, 1] with those moments. With three constraints (mass, mean, variance) the extreme points have at
most 3 atoms; we enumerate all 3-atom distributions on a 201-point grid and compute ln E e^{−R/λ} in log space
(`experiments/moment_worst_case.py`; an LP solver failed because the objective, ~1e-18, was below its tolerance).

μ = 0.9, δ = 0.05:

| n | σ = 0.01 | 0.03 | 0.04 (observed median) | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| 256 | 0.976 | 0.968 | 0.962 | 0.954 | 0.885 | 0.595 |
| 1,347 | 0.932 | 0.925 | 0.918 | 0.910 | 0.837 | 0.541 |
| 60,000 | 0.901 | 0.895 | 0.886 | 0.880 | 0.806 | 0.506 |

The worst case piles mass just below the mean (≈ μ − σ²/(1−μ)) plus an atom at risk 1. At the observed σ ≈ 0.04 the
bound is worse than the free 0.9. Reason: two moments only upper-bound tail mass (Chebyshev/Cantelli); lower bounds
(Paley–Zygmund) only reach fractions of the mean; going further requires all moments, i.e. the MGF = free energy.
The worst-case bound is monotone in σ at fixed μ (numerically), a *minimax* justification for ranking by variance,
not a usable certificate. A proof of monotonicity is open.

## 5. How much prior mass do published certificates imply? [ours, derived from published numbers]

If a posterior concentrates on a region A, KL(Q‖P₀) ≳ −ln P₀(A), so e^{−KL} lower-estimates the prior mass of the
region the posterior occupies. Valle-Pérez et al.'s −ln P(U) is the mass of all zero-error functions directly
(inverted from their realizable bound −ln(1−ε) ≤ (ln 1/P(U) + ln(2m/δ))/(m−1), δ = 0.05 assumed).
Table and sources: literature.md §"Prior mass implied by published certificates". Summary: data-free priors pay
0.04–0.19 nats per training example on MNIST (0.66 on CIFAR); a learned prior pays ~0.0001. A 1.5% kl-certificate at
n = 60k needs ≲ 0.015 nats/example from zero training error.

The 10⁻¹² requirement quoted by Picard-Weibel et al. is specific to Catoni's bound (Hoeffding-type penalty), target
1.5%, n = 60k; kl-type and realizable bounds need roughly ln(1/P) ≲ n·kl(train err ‖ target) (our approximation).

## 6. Catoni localization as thermodynamic integration [standard identity, our reading]

With the localized prior π_β ∝ e^{−βL}P₀: KL(Q‖π_β) = KL(Q‖P₀) + β·Q[L] + ln P₀(e^{−βL}).
For the Gibbs posterior Q̂_λ ∝ e^{−λR̂}P₀ and the empirically tilted prior (before the population/empirical swap),
with Z_t = P₀(e^{−tR̂}) and d ln Z_t/dt = −π_t(R̂):

- standard prior: KL(Q̂_λ‖P₀) = ∫₀^λ [π_t(R̂) − π_λ(R̂)] dt
- localized prior: KL = ∫_β^λ [π_t(R̂) − π_λ(R̂)] dt

Localization deletes the segment [0, β], where the Gibbs-mean risk falls from chance and where the ~10^{−2000} prior
mass lives. The swap from population to empirical tilt costs a concentration term and needs the Gibbs measure at β
to generalize. Differentiating once more, dπ_t(R̂)/dt = −Var_{π_t}(R̂), so the effective dimension at temperature t is
d_eff(t) ≈ 2t²·Var_{π_t}(R̂) (a heat capacity). **The prior variance (t = 0) is evaluated at the wrong end of the
curve**: it measures the segment localization removes. This explains why init-variance proxies behave as
trainability detectors.

**Why deep-learning PAC-Bayes does not localize.** With Gibbs posteriors the ln Z terms partly cancel. With Gaussian
variational posteriors (Pérez-Ortiz et al., García-Pérez et al.) nothing cancels and an upper bound on ln Z_β is
needed; ln Z_β ≤ 0 makes the bound slightly *worse* (factor 1 − β/λ on Q[L]). A small-β loophole [ours, rough]: for
β ≲ 1/σ the cumulant expansion of ln Z_β in the *population* mean/variance (estimable without a split, because prior
samples are independent of S) gives a certified gain of order β(μ − Q[L]) − β²σ²/2 ≲ 20 nats. Negligible at
KL ≈ 6,500 (MNIST-scale), 3–7% at KL ≈ 300–600 (CIFAR-scale). Needs a relative-error MC bound on Z_β to be rigorous.

## 7. Function-space vs. parameter-space KL at small σ [standard DPI + our argument]

Data processing: KL_m (logits on m inputs) ≤ KL_function ≤ KL_param. For narrow Gaussians, the push-forward of
N(μ, σ²I) is ≈ N(f(μ), σ²JJᵀ). With a shared Jacobian, every direction J sees carries the same KL in both spaces
(mean shift and both widths scale by the same singular value), so function-space savings at small σ come only from
(a) exact invariances (null space of J over all inputs), (b) nonlinearity along a large mean displacement, (c) the
many-to-one argmax (labels). Linearizing P and Q at their own means violates data processing (the ill-conditioned
prior covariance amplifies Jacobian differences); we observed KL_m > KL_param that way and discarded it.
Results: experiments.md, E-fkl.

## 8. Invariance-aware KL [ours; standard reparameterization argument]

If a group of weight rescalings g_t leaves every classifier unchanged, then Q and Q_t = (g_t)#Q define the same
Gibbs classifier, so a certificate may use min_t KL(Q_t‖P). This is valid for any data-dependent choice of t
(posteriors are arbitrary). Groups used (log-scales t):
- plain ReLU MLP: per hidden unit, incoming column and bias × e^{v}, outgoing row × e^{−v};
- LayerNorm: per layer, (W, b) feeding the LN × e^{s}; plus per-unit ReLU scaling of (γ, β) against the next layer;
- BatchNorm (reference-set or statistics-as-parameters): per unit, incoming column × e^{v}; for statistics as
  parameters also m_j × e^{v} and ln s_j + v (exact: (x·W_j − m_j)/s_j is invariant, no ε); plus the ReLU scaling.
LN/BN with ε>0 are only approximately invariant, so we always evaluate the empirical risk *with samples from Q_t*;
every reported certificate is rigorous for the classifier it describes.
**Finding:** training with t as free parameters lowers the KL by 30–65%; minimizing over t after plain training saves
<1%. The gain is an optimization effect (a better path to a lower-KL posterior), not removal of wasted nats.

## 9. BatchNorm under the i.i.d. assumption [ours]

Problems: training-mode BN couples examples within a batch; inference-mode running statistics make the map
weights → classifier depend on S, so a data-free prior over weights is not a data-free prior over classifiers.
Workarounds:
1. **Statistics as parameters (preferred).** The inference statistics (m_j, ln s_j) are posterior parameters with a
   data-free prior (N(0, 0.5²)). The classifier is a fixed per-example network, so the standard i.i.d. bound applies.
   Training procedure (any is allowed): batch statistics in phase 1, then freeze into the posterior with trainable
   mean and sd and fine-tune in inference mode so the statistics see the weight noise. The naive variant (statistics
   frozen after training, fixed sd) is markedly worse.
2. **Reference-set BN** (virtual BN, Salimans et al. 2016): statistics recomputed per sampled weight vector on a fixed
   set of unlabeled inputs held out from S. Valid and keeps BN's adaptation to weight noise, but needs held-out inputs.
Training-mode coupling within a batch only affects the *algorithm*, never the bound.

## 10. Block estimator of the Gibbs empirical risk [ours; novelty not established]

The certificate needs L̂_S(Q) = E_{θ∼Q}(1/n)Σᵢ ℓ(θ, zᵢ), estimated by sampling. **Standard (Langford & Caruana):**
draw θ₁..θ_M, evaluate each on all of S; each full pass is one independent [0,1] variable, so
kl(est ‖ L̂_S(Q)) ≤ ln(2/δ′)/M. Precision depends on M only, hence the 150,000 samples used in the literature.

**Block estimator.** Partition S into blocks of b examples (b divides n) and give each block its own fresh θ ∼ Q.
Block error rates X_j ∈ [0, 1] are independent given S, and (1/J)Σ_j E[X_j] = L̂_S(Q) exactly (linearity, equal
block sizes). Hoeffding's (1963) kl-Chernoff bound holds for independent [0,1] variables with different means (using
the average mean), so kl(X̄ ‖ L̂_S(Q)) ≤ ln(2/δ′)/N with N = passes·n/b. The partition must be independent of the θ
draws (random split). One pass costs one full evaluation but yields n/b terms.

At Gibbs error 0.15, n = 55,000, δ′ = 0.01: 150 full samples → bound 0.260; 150,000 full samples → 0.153;
blocks of 32, 20 passes (N ≈ 34k) → 0.156; blocks of 1 → 0.151. Our implementation uses b = 40 (divides 55,000;
N = 27,500 over 20 passes) and b = 250 for reference-set BN.
Literature check (2026-11-19): a quick search found only the full-sample scheme (Langford & Caruana; Pérez-Ortiz et
al.; García-Pérez et al.). A thorough search is needed before claiming novelty.

## 11. SGD-noise (subspace) posterior [ours]

Near a minimum, SGD's stationary covariance solves the Lyapunov equation HΣ + ΣH = (η/B)·C (Mandt et al. 2017). If
the gradient-noise covariance C ≈ H, then Σ ≈ (η/2B)·I on range(H) and 0 elsewhere: SGD spreads isotropically over
*curved* directions at temperature η/2B and never visits flat ones. So SGD's noise identifies the data-constrained
subspace, and its complement is where the data put no constraint.

Posterior in prior-whitened coordinates z = (θ − m₀)/s (s = σ₀ for weights, 0.5 for BN statistics), prior N(0, I):
Q = N(z̄, α·V diag(λ) Vᵀ + τ²(I − VVᵀ)), V, λ = top-K principal directions/variances of T SGD iterates.

KL(Q‖N(0, I)) = ½[α Σλ + τ²(P−K) + ‖z̄‖² − P − Σ ln(αλ) − (P−K) ln τ²].

Mechanism: with mean-field posteriors the KL is 92–99% mean term ‖μ − m₀‖²/2σ₀², and σ₀ is pinned small because the
posterior must be narrow in every coordinate. Here the complement width τ can be large (τ = 1 is free), so a larger
σ₀ becomes affordable if the network tolerates prior-sized noise outside the SGD subspace, which shrinks the mean term
∝ 1/σ₀². σ₀ comes from a fixed grid with a union bound (ln|grid| nats); K, α, τ are free posterior choices. The
point-estimate analogue of the KL mean term is an L2 pull toward m₀ during SGD.
Risks: the Hessian bulk is small but non-zero over ~10⁵ directions; the certificate covers samples from Q, not the SGD
iterate. Results: experiments.md, E-sgd.

## 12. Where PAC-Bayes certificates lose (decomposition) [ours, numerical]

PAC-Bayes-kl, n = 60k, stochastic train error 0.05: KL/n = 0.107 (data-free MNIST, Pérez-Ortiz) → 0.214 (theirs:
0.2165); deleting the confidence term ln(2√n/δ)/n = 0.00016 → 0.214; Pinsker/Hoeffding form → 0.282; KL 10× smaller →
0.089; learned prior (KL/n = 0.0001) → 0.055; KL/n = 0.107 with zero train error → still 0.10.
Conclusion: concentration (kl inversion) is nearly tight for the 0-1 loss; the slack is the complexity term (prior and
posterior family). Further limits: with the best data-distribution-dependent prior the expected KL is I(S;W);
Livni & Moran (2020) show PAC-Bayes cannot certify some learnable classes (thresholds), a limitation of the KL term
itself; conditional MI (Steinke & Zakynthinou 2020) avoids it.

## 13. Trajectory-based certificates: online-to-batch and online PAC-Bayes [standard results; our reading]

**Widening the posterior along the trajectory does not help with a Gaussian prior.** For any Q,
KL(Q‖N(m₀, σ₀²I)) = E_Q‖θ − m₀‖²/2σ₀² − H(Q) + const. Widening costs tr Cov/2σ₀² (linear) and gains ½ ln det Cov
(logarithmic); the optimal width in any direction is σ₀, which our best posteriors already use (τ = 1). The mean term
(thousands of nats) lies entirely along Δ = μ − m₀; spreading along the path only helps by including points closer to
m₀, which have higher loss (the trade-off the pull-to-init already tunes). A mixture over T trajectory points gains at
most I(t; θ) ≤ ln T nats.

**Progressive validation / online-to-batch** (Blum, Kalai & Langford 1999; Cesa-Bianchi, Conconi & Gentile 2004). With
one pass over S in a fixed order, X_i = ℓ(θ_{before z_i}, z_i) has E[X_i | F_{i−1}] = risk of that iterate. For [0,1]
martingale differences the Chernoff-kl bound holds for the average (the conditional log-MGF ln(1 − μ + μeᵗ) is concave
in μ), so the "random iterate of a window W" predictor satisfies kl(X̄_W ‖ μ_W) ≤ ln(1/δ)/|W|. No prior, no KL. It is
the sharpest form of Picard-Weibel's point (a test bound that recycles each example as training data after scoring it)
and certifies a single-pass learner only.

**Online PAC-Bayes** (Haddouche & Guedj 2022, Thm 2.3). Priors P_i may depend on z₁…z_{i−1} (an "online predictive
sequence"); posteriors Q_i may depend on all of S; the certified quantity is Σᵢ E_{Q_i}[E(ℓ | F_{i−1})] with slack
Σᵢ KL(Q_i‖P_i)/λ + λm/2 + ln(1/δ)/λ (Hoeffding-type). Consequences:
- Q_i = P_i (OPBTest) recovers progressive validation.
- Q̂_{i+1} = posterior after z_i, prior Q̂_i (OPBTrain): the KLs sum to a "step energy"; for Gaussians around SGD
  iterates, b·Σ_j ‖θ_j − θ_{j−1}‖²/2σ² per pass.
- **Multiple passes**: a constant multi-epoch posterior Q against first-pass priors pays KL(Q‖P_i) *per example*,
  giving slack ≈ √(2·avg KL) instead of √(KL/2n): vacuous for any realistic KL. Later passes are not forbidden, but are
  charged in full. SGD's minibatch noise does not help unless it is part of the prior/posterior (KL ∝ shift²/σ²): the
  quantitative versions of "noise makes data reuse cheap" are differential privacy (Dziugaite & Roy 2018) and
  information-theoretic trajectory bounds (Neu et al. 2021).
