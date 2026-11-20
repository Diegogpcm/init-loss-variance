# Research log

Chronological record of questions, decisions and dead ends. Dates absolute.

## 2026-11-19 (before this log): v1
NumPy MLP testbed on digits; CE-loss variance proxies with a PAC-Bayes free-energy motivation; E1–E3 (150-arch
sweeps, gain sweep). Found: raw CE variance dominated by logit scale; temperature-normalized between-init variance
ρ ≈ +0.4 but redundant with CVU. The PAC-Bayes framing in the v1 README was written by Claude in an earlier session;
the author (D.G.-P.) reports having had an independent PAC-Bayes idea, not yet written down here.

## 2026-11-19: v2 and the turn to PAC-Bayes certificates

1. **Setup on kipchoge.** Repo at `~/GitHub/init-loss-variance`; pip-less venv workaround; OpenML download stalled,
   Fashion-MNIST taken from the Zalando GitHub IDX files instead.
2. **Accuracy variance (author's request).** Added the 0-1 loss family, random-label controls, temperature-optimized CE
   free energy; full 1,200-arch grids with K = 128 on digits and Fashion. 0-1 variance beats CE variance; lower tail
   (best-of-K) beats variance; signal label-dependent.
3. **Literature.** Read Valle-Pérez et al. (2019, 2020) and Picard-Weibel et al. (2025) in full. Valle-Pérez: no new
   bound in 2019 (existing realizable bound + NNGP/EP), minor variant + optimality theorem in 2020. Picard-Weibel: no
   new bound; risk-prior functional (Thm 1), stochastic-order monotonicity (Thm 2), Catoni quantile requirements.
4. **Variance vs tail.** Author: symmetry ties tail and variance (mean accuracy is exactly 1/C). Confirmed (ρ ≈ 0.8) with
   residual shape information. Variance is not monotone with the bound in general; the convex order is (theory.md §3).
5. **Can moments give a stronger bound?** No: worst case over (μ, σ²) is worse than the free 0.9 at observed σ
   (theory.md §4). LP solver failed numerically (objective ~1e-18 below tolerance); replaced by exact enumeration.
6. **How much prior mass do priors put on good models?** Derived from published certificates (literature.md table):
   data-free priors 10^{−600} to 10^{−3700}; learned priors ~10^{−1}. 10^{−12} is specific to Catoni's bound.
7. **Author: need to incorporate the data distribution.** Localization (Catoni; Lever et al.) discussed; my held-out
   "train/test coherence" diagnostic was **rejected**: for prior samples the training set already is a test set, so it
   would show only sampling noise. Localization ⇔ deleting the [0, β] segment of the thermodynamic integral; the init
   variance measures that deleted segment (theory.md §6). Neither Pérez-Ortiz et al. nor García-Pérez et al. localize.
8. **Where do PAC-Bayes bounds lose?** Concentration nearly tight; slack is in the complexity term (theory.md §12).
   I initially ranked "parameter- vs function-space KL" first; the author predicted little difference at small σ.
   **E-fkl confirmed the author**: function-space KL recovers 75–85% of KL_param through 160 inputs at σ₀ = 0.02. My
   ranking was revised: gains must come from priors and posterior families.
9. **Author: PAC-Bayes should learn from practice (SGD, normalization).** Proposed invariance-aware priors and
   SGD-aware posteriors.
10. **Normalization (E-norm).** Author asked to incorporate BN despite the i.i.d. problem. Built reference-set BN and
    statistics-as-parameters; **the author prefers statistics as parameters** (no held-out data). Two-phase bn_stat
    with exact invariance gives the lowest KL. Invariance-aware *training* (not post-hoc rescaling) is the effective
    lever. BN slightly best in ordinary training too.
11. **SGD-noise posterior (E-sgd).** Lyapunov argument: SGD noise is isotropic over curved directions, absent in flat
    ones; posterior = SGD subspace (K directions) + isotropic complement; allows larger σ₀. Implemented with σ₀ grid
    union bound.
12. **MC estimation slack found.** Author asked why runs were so fast; checking revealed M = 150 posterior samples
    inflate certificates (correction 0.035 nats/example → +0.11 at Gibbs error 0.15) relative to the 150k samples used
    in the literature. Designed the **block estimator** (theory.md §10): fresh sample per block of b examples,
    N = passes·n/b independent terms; ~150k-sample precision for 20 passes. E-norm and E-sgd rerun with it; old
    results archived as `*_m150.jsonl`.
13. **E-sgd result (block estimator):** the SGD subspace adds nothing; the best posterior is the prior translated to
    the SGD average (τ = 1); the L2 pull to init is what matters (best certificate 0.374 vs 0.497 with M = 150).
    bn_stat suffered from sharing τ with the BN statistics (design flaw, to fix). I had wrongly reported the first
    SGD run as "stuck at 28 of 30": the grid has 28 configurations; both runs had completed.
14. **Normalization rerun (block estimator):** certificates ≈0.10–0.12 lower than with M = 150; bn_stat + inv best in
    3 of 4 settings (0.394–0.414).
15. **bn_stat fix in the SGD posterior** (own statistics width): 0.502 → 0.386 (width 64). New observation: with BN,
    certificates improve strongly with SGD temperature lr/B (ρ = −0.90) while SWA test error worsens slightly.
16. **Autonomous iteration (author away ~1 h, 2026-11-19 ~18:20):** round 1 = pull-to-init sweep {1e-2, 3e-2, 1e-1} ×
    {none, bn_stat} × width {64, 128} × 3 seeds with a new hybrid stage (SGD warm start + 10 epochs of direct bound
    optimization); variational seeds 1, 2 for inv-mode leaders.
17. **Round 1 result:** best certificate 0.355 ± 0.002 (prior translated to SGD mean, pull 3e-2, width 64); the hybrid
    collapses to the variational regime → hypothesis: the McAllester surrogate over-penalizes KL relative to the
    kl-inversion certificate. Round 2 tests f_quad; round 3 prepared: exact kl-inverse objective (implicit-function
    gradients, checked against finite differences).
18. **Round 2b crash:** a high-temperature bn_stat run diverged to NaN and the SVD exception killed the pool. Fixed:
    divergence is recorded as a result; each configuration runs in a try/except wrapper. Relaunched 18:39.
19. **Author's question (18:50): training on a CE bound to certify accuracy?** Yes: all objectives so far used the
    bounded CE as the empirical-risk surrogate. Implemented the author's method (García-Pérez et al. 2025, Sec. 4.1,
    eq. 9) as objective `klinv01`: kl⁻¹ evaluated at the rolling 0-1 risk, gradient m·∇L_xe with
    m = rolling L_01 / rolling L_xe (straight-through). 2-epoch smoke test: f_quad 0.440, exact kl⁻¹ on CE 0.435,
    klinv01 0.429 (higher KL, lower 0-1 error). Round 3 queued to compare at scale.
20. **Round 2 result:** f_quad beats McAllester by 0.011–0.016 everywhere (variational bn_stat 0.379); hybrid with f_quad
    gives bn_stat 0.365 (better than its SGD posterior and pure variational); best overall still the translated prior
    for plain networks (0.354). Higher SGD temperature hurts once the pull is strong.
21. **Round 3 (partial, 19:16):** the author's accuracy objective (klinv01) gives the best certificates: 0.347 ± 0.001
    (bn_stat hybrid) and 0.348 ± 0.001 (bn_stat variational alone). Objective ladder for variational bn_stat: 0.395
    (McAllester) → 0.379 (f_quad) → 0.364 (exact kl⁻¹, CE) → 0.348 (accuracy objective). Plain networks do not benefit
    from klinv01. Session closed by the author before the width-128 variational runs finished.
22. **Documentation for a paper (author's request):** `docs/` created (theory, literature, experiments, this log,
    paper outline).

## Decisions and preferences
- Init-variance-as-NAS-proxy: deprioritized (2026-11-19).
- BN in PAC-Bayes: statistics as parameters preferred over reference sets (no held-out data).
- Data-dependent priors via sample splitting: avoided (author's paper excludes them; Picard-Weibel critique).

## 2026-11-19 evening
23. **Round 3 complete.** Author asked whether the implicit (exact kl⁻¹) optimization makes f_quad and similar
    obsolete. Agreed for training objectives (exact kl⁻¹ ≥ f_quad in 8/8 settings, 7 strictly), with two caveats: the
    0-1 target (eq. 9) helps only for BN-statistics models and hurts plain networks (KL explodes); "optimal" holds for
    the function being optimized, not for the stochastic, non-convex optimization or for bounds outside PAC-Bayes-kl.
    First hypothesis ("plain networks lack the invariance-aware rescalings") was wrong: the plain-network runs were
    already in inv mode (ReLU per-unit rescaling). Refined hypothesis: BN-statistics models have rescalings that are
    free (no compensation in the next layer) and statistics parameters that absorb mean/scale shifts cheaply, so
    reducing the 0-1 error costs less KL; plain networks pay for the same functional change in full.

24. **Author: f_quad obsolete? / exploit low loss along the SGD trajectory?** Agreed to drop f_quad/McAllester as
    training objectives. Trajectory idea: a wider posterior along the path cannot help with a Gaussian prior (KL =
    second moment − entropy; optimal width = σ₀, already used). Read Haddouche & Guedj (2022) in full: priors may
    follow the past (online predictive sequence), posteriors arbitrary, per-step KL. E-online: progressive validation
    certifies 0.148–0.159 (2.3× tighter than our best PAC-Bayes, but a disguised test bound for a one-pass learner);
    OPBTrain with Gaussian noise around SGD iterates is vacuous (per-step energy). Author's question "does SGD noise
    keep later passes valid?": not by itself; later passes are charged in full by the per-example KL; noise must be
    part of the prior/posterior (DP / MI bounds) to make reuse cheap.

## 2026-11-20 (night, autonomous)
25. **DP-SGD + mutual information (author: "implement it to see what happens"; compare with other papers).** Derived a
    certificate for the trained network itself (theory.md §14): oracle prior = output marginal, disintegrated
    PAC-Bayes-kl, Markov on the information density, per-example MI chain rule with Poisson-subsampled Gaussian KL
    (I ≈ T/(2ν²n)). Rounds 1–3 on Fashion: 0.335 → 0.330 → **0.2745** (linear on data-free Gabor features). Key
    regularities: batch size irrelevant at fixed (T, ν); tiny gap; small models and features win under noise. Found
    Lampert & Zakerinia (arXiv:2605.26222, May 2026): high-probability max-information for DP-SGD, used as a PAC-Bayes
    prior (MNIST 0.29); no subsampling amplification; closest prior work. Pre-registered transfer to MNIST:
    **0.099 ± 0.004**, below the published data-free MNIST certificates we know (Lotfi 0.116). Gaussian PAC-Bayes on
    MNIST: bn_stat 0.228, plain 0.267, Pérez-Ortiz-style baseline 0.349. Mistake: `pkill -f` again killed its own ssh
    session (pattern matched the ssh command line); use `tmux kill-session` only.

26. **Verified comparison numbers and CIFAR-10.** García-Pérez et al. 2025 (HTML): MNIST best 0.315 (3×600 MLP,
    KL-modulated), CIFAR-10 best 0.738 ± 0.007 (4-layer CNN). Our Gaussian pipeline on MNIST (bn_stat + inv + klinv01,
    width-64 MLP): 0.228 ± 0.002, i.e. "their framework + our changes" improves by 0.09. DP/MI recipe A transferred
    unchanged to CIFAR-10: 0.740 ± 0.005 (tie). CIFAR download: Toronto mirror stalled; used the Hugging Face parquet
    (uoft-cs/cifar10), converted to channel-first rows in /tmp/lv_cifar10_raw.npz.

## Next steps (as of 2026-11-20)
0. DP/MI route: (a) high-probability tail for the information density with amplification (removes the 1/δ₁ = 42
   factor; worth ~0.03 on MNIST, ~0.05 on Fashion); (b) CIFAR-10 with data-free features (ScatterNet) against
   García-Pérez et al. 2025 and Lampert & Zakerinia; (c) hybrid: DP-SGD output as the *mean* of a Gaussian prior
   centred by MI + small posterior KL (Lampert & Zakerinia's structure, with our amplified MI); (d) verify the published
   MNIST numbers we compare against (Pérez-Ortiz FCN, García-Pérez MNIST) from the papers.

## Older next steps (as of 2026-11-19 evening)
1. Diagnose klinv01 on plain networks: track m, the 0-1/CE gap and KL during training; compare where the KL goes
   (per layer, mean vs variance terms) for none vs bn_stat under klinv01; try LayerNorm (per-example, has a free
   per-layer rescaling) with klinv01 to separate "normalization" from "statistics parameters".
2. Accuracy objective for LN and reference-set BN, and with σ₀ ∈ {0.1}; longer training.
3. Why do plain networks not benefit from klinv01 while bn_stat does? (KL blows up to 6,500; check the m ratio and
   the rolling average.)
4. Move to standard benchmarks (MNIST, CIFAR-10 CNNs with BN), GPU, with the author's code as baseline.
5. Literature checks listed in paper_outline.md (BN in PAC-Bayes, block estimator novelty).
