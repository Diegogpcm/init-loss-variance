# Paper outline (working draft)

Candidate framings, ranked by current evidence. Update as results come in.

## A′. Noise-aware certificates for trained networks (new lead candidate, 2026-11-20)

**Thesis.** SGD's noise, made explicit (DP-SGD), lets a certificate charge data reuse at ~T/(2ν²n) nats instead of
thousands of nats of parameter-space KL. Certify the trained network itself via MI + disintegrated PAC-Bayes
(theory.md §14). Headline: MNIST 0.099 ± 0.004 (recipe pre-registered on Fashion), Fashion 0.273 vs 0.347 for the best
Gaussian PAC-Bayes. Needed: tail bound for ı with amplification (the main technical contribution if it works), CIFAR-10,
careful comparison with Lampert & Zakerinia (2026), Lotfi et al. (2022), Dziugaite & Roy (2018). Threat: the certified
models are weak (DP-limited accuracy); reviewers may see it as "DP generalization" rather than PAC-Bayes for deep nets.

## A. Practice-aware PAC-Bayes certificates for neural networks (lead candidate)

**Thesis.** Data-free PAC-Bayes certificates for neural networks are loose mainly because of the complexity term,
not concentration (theory.md §12). Ingredients of practical deep learning, namely normalization, scale invariances and
SGD's noise geometry, can be built into the prior/posterior pair rigorously and reduce that term, without
data-dependent priors (which Picard-Weibel et al. criticise as disguised test bounds).

**Headline numbers (Fashion-MNIST, n = 55k, depth-2 MLPs, data-free prior, 3 seeds; 2026-11-19):** best certificate
0.347 ± 0.001 (BN statistics as parameters + invariance-aware KL + García-Pérez accuracy objective; hybrid with SGD warm
start), 0.348 ± 0.001 without SGD. Same family with the McAllester surrogate: 0.395. Plain network, same pipeline: 0.348
(hybrid) / 0.398 (variational).

**Contributions (status).**
1. *BatchNorm under the i.i.d. assumption*: inference statistics as posterior parameters with a data-free prior,
   two-phase training; exact scale invariance of (W_j, m_j, s_j) (theory.md §9). Status: implemented; lowest KL in
   E-norm (pending block-estimator rerun).
2. *Invariance-aware KL*: certify min_t KL(Q_t‖P) over weight rescalings that leave the classifier unchanged; training
   the log-scales jointly cuts KL by 30–65%; post-hoc minimization does not (theory.md §8). Status: implemented,
   needs ablations (is it an optimizer-path effect? learning-rate controls).
3. *SGD-noise subspace posteriors*: Lyapunov argument → posterior = SGD subspace + isotropic complement (theory.md §11).
   Status: **not supported** by E-sgd (isotropic posterior is as good). What did work: SGD with an L2 pull to the prior
   mean + posterior = prior translated to the SGD average (0.374 on Fashion MLPs). Needs: comparison with direct bound
   optimization under the same estimator (E-norm rerun), a sweep of the pull strength, and a literature check (this is
   close to Dziugaite & Roy's init-centred setup).
4. *Block estimator* of the Gibbs empirical risk: ~150k-sample precision at the cost of ~20 evaluations
   (theory.md §10). Status: implemented; novelty to be checked thoroughly.

**Missing before writing.**
- Standard benchmarks: MNIST and CIFAR-10 with CNNs (BN after conv layers), multiple seeds, GPU runs.
- Baselines on identical setups: Pérez-Ortiz et al. objectives and the author's own (García-Pérez et al. 2025, code at
  github.com/Diegogpcm/pacbayesgradients); same δ, same MC estimation for all.
- Ablations: invariance training vs. tuned learning rates; bn_stat phase lengths; SGD subspace K, T, iterate window;
  σ₀ grid size.
- Literature checks: prior PAC-Bayes treatments of BatchNorm; scale-invariant priors (e.g. on the sphere); per-sample /
  per-block MC estimation of Gibbs risk; SWAG-based PAC-Bayes posteriors.
- Accounting audit: every data-dependent choice must be either a posterior choice (free) or covered by a union bound
  (prior choices: σ₀ grid). Selection on training data is fine; certificates use fresh samples.

**Threats.** Gains may shrink with tuned baselines; certificates on Fashion-MNIST MLPs (≈0.5) are far from
state-of-the-art MNIST numbers; the SGD posterior certifies samples from Q, not the SGD solution.

## B. The prior's risk distribution and architecture quality (analysis paper)

**Thesis.** The init distribution's risk distribution is the object PAC-Bayes depends on (Picard-Weibel Thm 1). We
study what its statistics can and cannot say: 0-1 symmetry pins the mean; convex order (not variance) orders bounds;
two moments certify nothing (moment worst case); the init variance measures the segment of the free-energy curve that
Catoni localization removes, which explains why init-variance proxies act as trainability detectors (theory.md
§§2–6); audit of the prior mass implied by published certificates (literature.md). Empirical: 4,920-arch grids
(E-v2), tail vs variance (E-v2-tail), CVU as the 0-1 analogue.

**Status.** Results exist; the predictive signal is modest (ρ 0.2–0.57) and mostly redundant with CVU, so this works
best as an analysis/negative-results paper or as background sections of A. Missing: NAS-Bench-201 / NATS-Bench (to test
CVU's sign flip), heat-capacity measurements at relevant temperatures (MCMC), nested sampling / AIS of the risk-prior
curve for small MLPs.

## C. Short note: block estimator for Gibbs-risk certificates

Self-contained lemma plus numbers (theory.md §10) if the literature check confirms novelty. Could be an appendix of A.

## Open questions worth a section
- Proof that the moment worst-case bound is monotone in σ at fixed μ.
- Convex-order monotonicity for the whole class B(D, b) of Picard-Weibel et al.
- Rigorous small-β localization gain with a relative-error MC bound on Z_β.
- Does the certificate track learning-rate / batch-size effects on test error (Valle-Pérez desideratum D.4)? (E-sgd
  algorithm axis.)
- The author's own PAC-Bayes idea (not yet recorded here).
