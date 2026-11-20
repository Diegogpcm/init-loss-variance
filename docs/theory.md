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

## 14. Certifying a multi-epoch network trained with noisy SGD, via mutual information [derived here from standard pieces]

Goal: make "SGD noise makes data reuse cheap" quantitative for a *single, deterministic-at-test-time* network trained for
many epochs, without a held-out set and without a posterior around it. Algorithm A = DP-SGD (Abadi et al. 2016):
Poisson sampling with rate q = B/n, per-example clipping to norm C, G_t = Σ_batch clip(g_i) + N(0, σ²C²I), any
post-processing (momentum) of G_t. Data-free init.

1. **Oracle prior.** P* = marginal law of W_T over S ~ Dⁿ and A's randomness. It is data-free (unknown, but never
   needs to be evaluated). Q_S = law of W_T given S. Information density ı(W, S) = ln dQ_S/dP*(W), E[ı] = I(S; W_T).
2. **Disintegrated PAC-Bayes-kl** (change of measure on a single draw; Blanchard & Fleuret 2007, Rivasplata et al.
   2020): E_S E_{W∼Q_S} e^{n·kl(L̂_S(W)‖L(W)) − ı(W,S)} = E_S E_{W∼P*} e^{n·kl} ≤ 2√n (Maurer), so with probability
   ≥ 1 − δ₂ over (S, W): kl(L̂_S(W)‖L(W)) ≤ (ı(W,S) + ln(2√n/δ₂))/n. W is *the trained network*; no Gibbs estimation.
3. **Markov on the information density.** E[e^{−ı}] ≤ 1 and −x ≤ e^{−x−1} give E[ı⁻] ≤ 1/e, so
   E[ı⁺] ≤ I + 1/e and P[ı > (I + 1/e)/δ₁] ≤ δ₁.
4. **MI of DP-SGD, with the 1/n per example.** For a product distribution, I(S; Y) = Σᵢ I(zᵢ; Y | z_<ᵢ) ≤
   Σᵢ I(zᵢ; Y | z_−ᵢ) ≤ Σᵢ E KL(P_{Y|S} ‖ P_{Y|S∖zᵢ}) (the leave-one-out mechanism is a valid reference measure).
   With Y = (W₀, G₁…G_T), the adaptive chain rule gives a per-step KL of the Poisson-subsampled Gaussian,
   KL_step(q, σ) = KL((1−q)N(0,σ²) + qN(1,σ²) ‖ N(0,σ²)) (worst case at ‖g‖ = C by data processing:
   x ↦ (s/C)x + N(0, σ²(1 − s²/C²))). Hence I(S; W_T) ≤ n·T·KL_step ≈ n T q²/(2σ²) = epochs · B / (2σ²) =
   epochs / (2ν²B), with ν = σ/B the noise std on the *mean* gradient (units of C).
   Large batches are favoured at fixed ν.

**Certificate** (probability ≥ 1 − δ₁ − δ₂):
  kl(L̂_S(W) ‖ L(W)) ≤ [ (I_ub + 1/e)/δ₁ + ln(2√n/δ₂) ] / n.
We use δ₁ = 0.024 and δ₂ = 0.001, since the KL is linear in 1/δ₁ and only logarithmic in 1/δ₂. Comparison with the
Gaussian-prior certificates: here the "KL" budget is (I + 1/e)/δ₁ ≈ 42·I nats. It must be paid for every epoch, but it
does not grow with the distance travelled from init or with the parameter count; only the noise-to-batch ratio matters.

Notes.
- This is the information-theoretic counterpart of Dziugaite & Roy (2018) (DP prior + PAC-Bayes; they need *pure*
  ε-DP and use max-information, ε²/2 per example). Here only a KL/MI bound is needed, so Gaussian noise suffices with
  explicit constants.
- The Markov factor 1/δ₁ was the main loss; §14.1 replaces it with a high-probability bound (L-CL-Z Thm 8).
  The "no-Markov" column in the experiments plugs E[ı] in place of the tail and is *not* a certificate, only the
  limit of this route.
- Selection over a hyperparameter grid of K configurations must divide δ₁ by K (linear cost), unlike the ln K of the
  Gaussian-prior union bounds.

**Comparison with Lampert & Zakerinia (2026), v1.** v1 bounds the β-approximate *max-information* of DP-SGD directly
(high probability, no Markov), by Jensen against an independent copy S̃ of the whole dataset. That comparison sees the
full batch change (all m examples differ), so it cannot use subsampling amplification: the leading term is
n·E/(2σ²) (σ = noise multiplier), versus E·q·n/(2σ²) = epochs·B/(2σ²) for our per-example chain with Poisson
sampling, a factor q = B/n smaller. In their PAC-Bayes use the DP-SGD output is a prior and a Gaussian posterior still
pays KL ≈ 5,000.

### 14.1 The tail of ı with amplification: Lampert, Cairney-Leeming & Zakerinia (2026, v2), Thm 8 [theirs; applied here]

v2 of the same paper (arXiv:2605.26222v2) contains the high-probability bound we listed as the open lever. It works for
*any* mechanism that is (α, ε)-RDP under replace-one neighbours, so Poisson amplification enters through ε.
- Per-example filtration F_i = σ(Y, z_≤i). ı = Σᵢ ln rᵢ(zᵢ), where rᵢ is the posterior-to-prior density of zᵢ given
  (Y, z_<ᵢ), with the suffix z_>ᵢ marginalised.
- J_i = H_α(post‖D)·H_α(D‖post) = exp((α−1)·symmetrised Rényi) ≥ 1. Joint convexity of the Hellinger integral
  (the suffix is mixed with the same weights) plus replace-one RDP give E J_i ≤ e^{(α−1)ε}.
  This is the only place privacy is used.
- Let v_i = ψ(J_i − 1)/(2(α−1)²), with ψ(u) = asinh²√u. Backward integration under the product measure gives
  E exp(λı − b_λ Σv_i) ≤ 1, with b_λ = λ + λ²/(1 − λ/(α−1)) and λ < α − 1.
- The realised sum Σv_i is controlled by Jensen (ψ is concave) plus Markov on (1/n)Σ(J_i − 1), at level β₁.
  This is still a Markov step, but it sits inside ψ, which grows like ¼ln²u, so it costs a log factor instead of 1/β.

Result: P[ı > v⋆ + 2√(v⋆L₂) + L₂/(α−1)] ≤ β₁ + β₂, where v⋆ = n/(2(α−1)²)·ψ((e^{(α−1)ε} − 1)/β₁) ≤ (n/8)(ε + ln(4/β₁)/(α−1))²
and L₂ = ln(1/β₂).

Combined with step 2 above (disintegrated PAC-Bayes on the trained network, union with δ₂), the certificate becomes
kl ≤ (κ_LZ8 + ln(2√n/δ₂))/n, again with β = 0.024 and δ₂ = 0.001. κ_LZ8 depends only on (n, q, σ, T), so optimising
α and β₁ is free. Replacing Markov by it needs no union bound, because the choice is data-free.

Implementation: experiments/tail_bounds.py.
- Replace-one RDP is obtained from add/remove RDP, using the integer-order formula of Mironov, Talwar & Zhang (2019),
  and the weak triangle inequality as in their App. C: ε_rep(α) = (2α−1)/(2(α−1))·Tε_add(2α) + Tε_add(2α−1).
- We checked numerically that this conversion is within 1–3% of the directly computed replace-one divergence.
- We also checked numerically that the antipodal pair h = −h′ is the worst case over |h|, |h′| ≤ 1 in 2-D, so a
  direct replace-one accountant would gain nothing.

At recipe A (n = 60k, B = 4096, ν = 0.01, 10 epochs; I_ub = 12.2): κ_Markov = 522, κ_LZ8 = 185 (α ≈ 82, β₁ ≈ 0.02),
against the non-certificate E[ı] ≈ 12. Asymptotically κ_LZ8 ≈ 2·I_ub·ln(4/β₁) ≈ 11·I_ub, against 42·I_ub for Markov.

**Where the remaining slack is (κ_LZ8 ≈ 185 against I ≈ 12):**
- (a) *Replace-one versus remove.* Worst-case replace-one is antipodal, |h − h′| = 2, which is 4× the add/remove ρ.
  The quantity actually used, E_{z,z′} H_α(P_z‖P_z′), only needs E|h(z) − h(z′)|² = 2·Var h ≤ 2.
  So a factor ≈ 2 in v⋆ is lost to the sup. Recovering it would need an "average-pair" RDP that still composes
  adaptively; plain averages do not compose.
- (b) *The Markov step on Σᵢ(J_i − 1).* The realised posterior divergences share the same Y. If every example's
  gradient variation lies along one fixed direction at every step, Σᵢ(J_i − 1) ∝ G², with a single Gaussian G built
  from the summed noise, so it is χ²₁-like. A bound that uses only RDP therefore cannot drop the ln(1/β₁) factor
  entirely.
  In that aligned case, though, I_ub itself is very loose: the channel is effectively one scalar Gaussian, so the
  true I ≈ ½ln(1 + 2I_ub). For diverse gradient directions over T steps with fresh noise, the sum concentrates like
  χ²_{T·r_eff}.
  L&Z exploit exactly this step structure for *streaming* DP-SGD (their Thm 4, close to I + O(√(I·L)) + O(L)). There,
  batch t is fresh data, so the posterior predictive of step t is the population one. With multi-epoch reuse, the
  posterior predictive involves the posterior over S, whose cross-example correlations would have to be controlled.
  A Dobrushin-type route is open: the pair interactions are q²⟨h_i,h_j⟩/σ² per step, summed to ≤ 2I/n per pair.
- Ceiling at fixed hyperparameters (MNIST A): 0.084 (Thm 8) → 0.073 if κ = 2I + 2√(2IL), → 0.071 if κ = I + 2√(IL);
  the not-a-certificate E[ı] plug-in gives 0.069. Fashion: 0.246 → 0.230 / 0.227 / 0.224. CIFAR-10: 0.715 → 0.696 / 0.693 / 0.689.
  The Markov → Thm 8 step captured about 60% of the gap between the Markov certificate and E[ı].

### 14.2 A structured tail for multi-epoch Poisson DP-SGD: reduction, heuristic, measured value [ours; not a theorem]

**Reduction.** Thm 8 loses only at the Markov step on (1/n)Σᵢ(J_i − 1). The examples are i.i.d. from μ, so
J_i ≈ J(Y) for every i, and the n terms are essentially perfectly correlated. Concentration can only come from Y,
i.e. from the T fresh noise vectors.

**First-order structure.** The posterior log-tilt of a generic example is s(z) ≈ Σₜ (q/σ²)⟨wₜ, hₜ(z) − h̄ₜ⟩. Here wₜ
is the residual noise: Gaussian noise plus the Poisson noise of the other examples, with covariance
Σ_w = σ²I + B·E[hhᵀ]. So ln J(Y) ≈ c·wᵀAw. A ∝ Σ_w^{1/2} C Σ_w^{1/2}, where C is the covariance of the gradients
stacked over all T steps, under the *population* at the realised trajectory.

**Heuristic bound.** Replace Markov by Hanson–Wright / Laurent–Massart for this chaos:
- c·wᵀAw ≤ (α−1)ε·K, with K = 1 + 2√L₁·‖A‖_F/trA + 2L₁·‖A‖_op/trA, and the rest of Thm 8 unchanged.
- Implemented as `tail_bounds.struct_kappa_heuristic`.
- It beats Thm 8 only if ‖A‖_op/trA ≲ 0.25. Even at full diversity it gives κ ≈ 53 (MNIST about 0.074), because the
  replace-one factor and ψ remain.
- In the aligned worst case, K ≈ 1 + 2√L + 2L ≈ 13, which is worse than Thm 8.

**What a proof would need:**
- **S1. A rigorous chaos representation of ln J (the hard part).** It must handle the Poisson mixtures, the
  marginalised suffix and the conditioned prefix, uniformly over the tails.
- **S2. Hanson–Wright with predictable blocks.** The noise is fresh at each step, so a tower/MGF argument should work.
- **S3. Certifying the diversity ratios.** A is a population quantity at a data-dependent trajectory. Options: a
  training-set estimate transferred through the Thm 8 tail, P_joint[E] ≤ e^κ·P_prod[E] + β; or simply the worst case.

**Measured value** (`dp_sgd_bound.py --diag_m 1500`, recipe A, seed 0, spectrum of A estimated on 1,500 test points
at every step; `results/v2/dp_diag_*.jsonl`):

| | eff. rank | ‖A‖_op/trA | ‖A‖_F/trA | trC/T | Thm 8 | structured (heuristic) |
|---|---|---|---|---|---|---|
| MNIST | 17.7 | 0.135 | 0.238 | 0.37 | 0.0874 | 0.0837 |
| Fashion | 9.3 | 0.232 | 0.328 | 0.49 | 0.2482 | 0.2467 |
| CIFAR-10 | 20.8 | 0.100 | 0.219 | 0.78 | 0.722 | 0.714 |

The effective rank is about the number of classes, not T·rank. Each example's gradient direction persists across
steps, so the steps do not add independent directions.

**Verdict:** even if S1–S3 were proven, the gain at these hyperparameters is 0.002–0.008. Not worth pursuing.

**The larger remaining slack is worst-case sensitivity.** The accountant assumes every clipped gradient has norm C and
that the worst pair is antipodal, i.e. E|h − h′|² ≤ 4. The measured centred variance is trC/T ≈ 0.37–0.78 per step,
so an average pair has 2·trC/T ≈ 0.7–1.6.
- Plain DP cannot exploit this, because it is worst case by definition.
- Individual privacy accounting (Feldman & Zrnic 2021, Rényi filters) charges each example by its realised gradient
  norms and stops it at a fixed budget. That keeps the same replace-one ε, but allows *more steps* for the same
  certificate. It is the next lever to test, since training error still falls with epochs (Fashion ν = 0.01:
  0.214 → 0.200 from 10 to 20 epochs).

### 14.3 Individual privacy accounting (Rényi filter) [Feldman & Zrnic 2021, applied here; negative result]

**Construction.** Feldman & Zrnic (2021), Thm 4.5: if each point is charged, at every step, its individual RDP
ρₜ⁽ⁱ⁾, taken as the sup over the other points of the add/remove divergence and computed from the past and the point
itself, and is dropped once Σₜρₜ⁽ⁱ⁾ would exceed B, then the composition is (α, B)-RDP.
- For a Poisson-subsampled Gaussian step, ρₜ⁽ⁱ⁾ = ε_SGM(q, σ/rᵢ, α), with rᵢ the point's clipped gradient norm at
  θ_{t−1}. This needs every point's gradient norm at every step.
- Filtering at the two orders 2α and 2α−1 (their Remark 4.4) with budgets T₀·ε_SGM, then the weak-triangle
  conversion, gives replace-one ε_rep(α) equal to that of an unfiltered T₀-step run. So Thm 8's κ is unchanged while T
  can grow. Points with r = 0.5 get 4× as many steps.
- Implementation: `IndividualFilter` in dp_sgd_bound.py (`--filter_alpha 81.5 --filter_T0 146`); rescoring in
  `rescore_tail.kappa_filtered`.

**Result** (Fashion, recipe A, 10–40 epochs, lr 0.3/1, seed 0; `results/v2/dp_filter_fashion.jsonl`):
- No gain. Best filtered certificate 0.2512 (10 or 15 epochs), against unfiltered recipe A at 0.2499.
- Note: Fashion has n = 55k, so recipe A runs 134 steps. T₀ = 146 gave the filter a 9% larger budget (κ 201 against
  185), and it still did not win.
- Longer filtered runs have *higher* training error than unfiltered runs of the same length (20 epochs: 0.217
  against 0.200; 40 epochs: 0.238). By 40 epochs 17% of points have exhausted their budget and about 40% are inactive.
- Reason: the points whose budget runs out are the high-gradient, i.e. misclassified, points, which are exactly those
  the training error depends on. Points the model already fits cost little but no longer matter.
- Verdict: individual accounting does not help the training-error / κ trade-off in this regime.

**Selection.** Hyperparameter selection over K configurations on the same S costs δ₁/K (linear) on this route. Protocol
used: choose recipes on Fashion-MNIST, then apply them *unchanged* to MNIST (pre-registered transfer, no selection on
MNIST), and re-seed on Fashion.

## 15. Certifying a deterministic SGD-trained network: what SGD's own noise provides [diagnostics, ours]

Setting: a standard ReLU MLP, 784-256-256-10 (269k parameters), trained with plain SGD (lr 0.1, batch 128, 10 epochs,
no clipping, no noise) on Fashion. Train 8.6% / test 11.7%. Scripts: experiments/sgd_influence.py and sgd_cloud.py.
Results: results/v2/sgd_influence_fashion_w256.json and sgd_cloud_fashion_w256.json.

**15.1 Leave-one-out influence through the training dynamics.**
- Bound: perturb the output only in the analysis, W~ = W_T + N(0, diag σ²); then
  I(S; W~) ≤ Σᵢ E‖W_T(S) − W_T(S∖zᵢ)‖²_{diag σ⁻²}/2 (chain rule + Neu et al. 2021, Lemma 4).
- Influence linearised through Π(I − ηH_t) and estimated with Hutchinson probes, by backpropagating through training.
- Checks: the linearisation matches finite down-weighting at ε = 1e−4 to six digits.
- **Linear influence** Σᵢ‖ΔWᵢ‖² ≈ 31, against ‖W_T − W₀‖² ≈ 102: only about 3× smaller than the init-centred KL.
  Plug-in values stay vacuous (0.72 at the best width, against 0.97 for the init-centred KL).
- **Exact leave-one-out** (48 retrains, same batches) is dominated by chaos:
  - Every removal gives ‖ΔW‖ ≈ 1.6–1.9, with cosine ≈ 0 to the linear prediction.
  - A float64 replay of the *unchanged* run ends 2.4 away from the float32 run.
  - Coupled-seed stability bounds therefore cost about 3×10⁸ nats. Dead.

**15.2 Is SGD's randomness a free noise?** Same init; 16 batch orders on each of two disjoint halves A and B.
- Parameter space: per-cloud spread E‖W − mean‖² = 6.0; A–B centre shift 27.6; travel from init 47.6; init perturbed
  by 1e−6: 3.5; two batch orders: 11.3.
  The isotropic KL between clouds at their own scale is 6.5×10⁵, against 9.5×10⁵ init-centred: a factor of 1.5.
  **In parameter space, seed noise does not hide the data.**
- Function space (argmax disagreement / mean symmetric KL of the softmax outputs):

  | set | within seeds | between halves |
  |---|---|---|
  | test | 8.3% / 0.085 | 9.9% / 0.121 |
  | A's training points | 8.1% / 0.087 | 10.3% / 0.135 |

  **In function space, seed noise accounts for about 84% of the cross-dataset disagreement.** Data-specific variation
  adds only about 1.6–2.2 points of disagreement on top of what batch order alone produces. Within-seed disagreement,
  8.3%, is itself close to the generalization disagreement equality regime (Jiang et al. 2022: disagreement ≈ test
  error), against a test error of 12.8% here.

**Reading.** SGD's noise matters for generalization in function space, not in parameter space. The parameter-space
routes (stability, influence, Gaussian posteriors at the seed-cloud scale) cannot use it. The natural framework is
information in *predictions*: f-CMI / e-CMI (Harutyunyan et al. 2021; Hellström & Durisi 2022), where randomness from
the seed directly reduces the conditional information between the predictions on a supersample and the train/test
selection. Open issues for a *certificate*:
- those bounds are in expectation;
- their estimation needs supersample (ghost) data and many retrainings.

