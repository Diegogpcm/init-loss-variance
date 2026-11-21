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

27. **Tail bound with amplification (author: "work on the tail bound with amplification").** Found that v2 of
    Lampert et al. (arXiv:2605.26222v2, with Cairney-Leeming) already proves it: Thm 8 gives an RDP-only bound
    (replace-one; Poisson amplification enters through ε) via a posterior-to-prior filtration over examples and
    Markov *inside* asinh². Implemented it (`experiments/tail_bounds.py`) and rescored all runs (`rescore_tail.py`).
    - Recipe A: κ = 185 against Markov's 522.
    - Pre-registered transfers: MNIST 0.0995 → **0.0836**, CIFAR-10 0.740 → **0.715** (L-CL-Z's best CIFAR is 0.716,
      García-Pérez's 0.738), Fashion 0.273 → 0.249. The Fashion selection is unchanged.
    - Checks: the weak-triangle add/remove → replace-one conversion is within 1–3% of direct accounting; the antipodal
      pair is the worst case in 2-D.
    - Remaining slack, analysed in theory.md §14.1: (a) replace-one sup versus average pair, about 2×; (b) Markov on
      the realised Σ J_i, about ln(4/β₁).
    - (b) is unavoidable from RDP alone, because aligned gradients give a χ²₁-like sum. Removing it needs the Gaussian
      step structure, as in their streaming Thm 4, extended to data reuse; that is open.
    - Ceiling at fixed hyperparameters: MNIST 0.071–0.073, Fashion about 0.23, CIFAR-10 about 0.695.
    - Ruled out on the way, by our own analysis: I_α^Sibson(S;Y) ≤ Σᵢ sup I_α(zᵢ;Y|z₋ᵢ) is false for every α
      tested (1.5, 2, 3). Our counterexample: a "diagonal" channel where Y reveals the example index only when z₁ = z₂;
      at α = 2, k = 100 it gives 1.38 against 0.36. Per-example RDP is infinite there, so the RDP version is still possible.
      Converting RDP → (ε,δ) → the Rogers et al. max-information is worse than Markov at these parameters, because of
      the per-example ln(1/δ_DP).

28. **Structured multi-epoch tail (author: "try the structured multi-epoch tail bound").** Reduction (theory.md §14.2): Thm 8
    loses only through Markov on Σᵢ(J_i − 1). The J_i are essentially equal across i.i.d. examples, so concentration
    must come from Y. To first order, ln J(Y) is a Gaussian chaos c·wᵀAw, with A the noise-weighted population gradient
    covariance stacked over steps.
    - Heuristic bound (Hanson–Wright in place of Markov) in `struct_kappa_heuristic`.
    - Measured A on our trajectories (`--diag_m 1500`, test points): effective rank 9–21, op/tr 0.10–0.23, because
      per-example gradient directions persist across steps.
    - Even if the three missing lemmas (chaos representation, predictable Hanson–Wright, certified diversity) were
      proven, certificates would improve by only 0.002–0.008.
    - Decision: do not pursue.
    - The diagnostic shows a bigger slack in worst-case sensitivity: centred gradient variance trC/T = 0.37–0.78
      against the worst case's antipodal |h − h′|² = 4. Next lever: individual privacy accounting (Feldman & Zrnic 2021
      Rényi filters), which allows more steps at the same ε.

29. **Individual privacy accounting (author: "okay, try that").** Implemented the Feldman–Zrnic Rényi filter for Poisson
    DP-SGD at orders 2α and 2α−1, which keeps Thm 8's κ while allowing more steps (theory.md §14.3).
    - Fashion grid, 10–40 epochs: no gain. Best 0.2512 against recipe A's 0.2499.
    - The filter drops the high-gradient (misclassified) points first, so longer runs train *worse*
      (20 epochs: 0.217 against 0.200 unfiltered).
    - Bookkeeping mistake: on Fashion (n = 55k), T₀ should have been 134. The filtered runs had a 9% larger budget and
      still lost. Not transferred to MNIST/CIFAR.
    - Author also asked whether the minibatch randomness is credited. Own inclusion: yes (amplification). Others'
      inclusion: no; DP is worst case over them, and it is worth about 9–15% extra variance in the signal directions,
      ≈ 0.002. Others' data values: not credited.

30. **Scattering features (author: "try the scattering features on fashion, but i think that deviates too much from standard
    deep learning ... we would ideally be looking to explain why deep learning generalizes").**
    - Added second-order scattering (81 channels).
    - Recipe S was chosen on Fashion (27 runs) and transferred unchanged: MNIST **0.0695 ± 0.005** (test 4.1%; was
      0.084), CIFAR-10 0.716 (no change; test 0.660), Fashion 0.241.
    - Strategic agreement with the author: the DP/MI route explains generalization of noisy, information-starved
      training of fixed-feature models, not of standard deep nets. Plain SGD has I(S;W) = ∞ for continuous weights,
      and the same networks fit random labels.
    - Proposed main line: a **two-phase hybrid on standard architectures**.
      1. A noisy-SGD phase, charged by the Thm 8 κ (amplified, about 185–340 nats), gives a data-dependent prior mean
         with no held-out split.
      2. Ordinary training continues; a Gaussian posterior on the final standard network is charged by KL to that
         prior, using our pipeline (BN statistics as parameters, inv KL, klinv01).
      This is L-CL-Z's Table 3 structure, with κ ≈ 185 against their 1,224, plus our KL improvements. It certifies a
      standard network at near-standard accuracy and splits generalization into "cheap noisy learning" plus
      "flatness of fine-tuning".
    - The best-certificate track (small models, fixed features) is now secondary.

31. **Deterministic network via SGD noise (author: certify a deterministic network; what does SGD noise give; gradients
    against the population, taking architecture/gradient flow into account).** Diagnostics on a standard MLP trained by
    plain SGD on Fashion (theory.md §15).
    - (i) Linearised leave-one-out influence through the training Jacobians is only 3× smaller than the distance from
      init. Exact leave-one-out is swamped by chaos: float32/64 rounding alone moves the endpoint as far as removing an
      example. Parameter-space stability is dead.
    - (ii) Seed clouds (batch order) are small compared with the data-to-data shift in parameter space: KL ratio 1.5.
    - (iii) In function space, seed disagreement (8.3%) is about 84% of cross-dataset disagreement (9.9%).
    - Conclusion: SGD noise acts in function space. The matching literature is f-CMI (Harutyunyan et al. 2021),
      e-CMI (Hellström & Durisi) and disagreement-based estimators (Jiang et al. 2022).

32. **Read the f-CMI / e-CMI / single-draw / ld-CMI papers (author: "see if a certificate is possible ... use the
    chaotic nature of training to our advantage").** Verdict (theory.md §15.3):
    - Supersample bounds certify the ghost-half loss, which held-out evaluation already gives better. They are
      explanatory, not certificates.
    - Single-network certificates need an upper bound on an implicitly defined information (ı(S; f)), which samples
      cannot provide.
    - Chaos can enter a certificate only via provable mixing of an explicit-noise process (open, likely impractical).
    - Chaos is a strong *explanatory* lever: self-generated noise masking sample-specific signal in function space,
      measurable with finite-alphabet information.
    - Proposed experiment: f-CMI / ld-CMI on Fashion MLP for fixed vs random seed, chaotic vs chaos-suppressed training,
      true vs random labels.

33. **Chaos experiment on Fashion (author: "run the chaos experiment").** theory.md §15.4.
    - The first two calibrations had hung for hours: fork after torch init deadlocked. Fixed with spawn-started workers.
    - Chaos index (test flips under a 1e-6 init change): SGD 1.0%; GD lr 0.3 13.8%, lr 0.1 0.7%, lr 0.03 0.17%.
      Full-batch GD is not chaos-free by default.
    - Main run, 640 trainings: fixed seed ≈ random seed within 3% for chaotic SGD *and* near-non-chaotic GD.
    - In the interpolating regime, ld-CMI is pinned to the ghost error, so self-generated noise cannot mask
      membership. ld bound ≈ 2.1× gen everywhere. SGD vs GD: SGD has ~20% less information, independent of seed
      (implicit bias, not noise).
    - Random labels: GD I = 0.62 nats ≈ 0.9·ln 2 (memorisation); chaos does not hide it.
    - Verdict: chaos-as-noise falsified in the CMI sense. A noise-based explanation must act on training points
      (not fitting some), which plain SGD does not do.

34. **Lyapunov / contraction route (author: "differential equation results like Lyapunov's ... synergize with
    PAC-Bayes?"; chose the time-uniform information route).** theory.md §16.
    - Implemented Altschuler & Talwar 2022's explicit privacy-amplification-by-iteration bound (strongly convex,
      weight decay) inside L-CL-Z Thm 8, plus a contractive trainer: no momentum, no clipping (normalised features
      give L = 1), projection.
    - Fashion grid (264 runs): best certificate 0.321, attained at pure composition, vs 0.249 for recipe A.
      The bound saturates κ only for T ≫ 1/(ηλ). The weight decay needed for fast forgetting caps train error at
      ≥ 0.33 (λ ≥ 0.01).
    - Structural reason: forgetting time ≈ optimisation time ≈ 1/(ηλ). The non-convex Doeblin constant is
      exp(Θ(D²/s²)).
    - Route closed. The Riemannian / pullback-metric posterior (route 3 of the discussion) remains.

35. **Prequential AIS certificate (author: "mix PAC-Bayes into existing ideas ... bounding the uncertainty").** theory.md §17.
    - Construction: data-tempered annealed importance sampling with Metropolis-adjusted HMC. Its extended-space
      log-ratio λ(prequential − final log-loss) is a valid disintegrated PAC-Bayes complexity for the single final
      network.
    - MNIST MLP, full 60k: certificate 0.217 (λ = 1), then 0.167 with geometric early stages and λ = 0.5 (test 0.040).
      Beats the Gaussian / data-free baselines (≈ 0.27–0.3); does not beat the DP-SGD certificate (0.084).
    - The limit is the single-sample learning curve, not MCMC compute.
    - Closest prior work: Excess Description Length (arXiv 2601.04728), in expectation only and for SGD.

## 2026-11-21

36. **AIS certificate: features, particles, validity, exactness (author: "get working on it"; "results look almost too
    good to be true ... check the validity of the math"; "will you figure out a way to bound the effect of [floating
    point]?").** theory.md §17.1–17.9, literature.md "Sampler-based certificates and floating-point exactness".
    - Levers, full 60k MNIST: pixels 0.167 → Gabor + MLP 0.0726 (two seeds, both 0.0726) → second-order scattering +
      MLP **0.0648** (test 0.015), below our DP-SGD best (0.0695). Fashion: **0.2207** (test 0.126), below DP-SGD's 0.249.
      SMC particles ≤ 3% of C; width 300 worse; λ ∈ [0.15, 0.5] flat.
    - Proof written out (Lemma 1 path identity; Theorem 2). The point a referee will check: the extended prior depends
      on S through the kernels; only its w_K-marginal, which is the data-free P, enters. Sanity bound
      E[C] ≥ KL(law(w_K | S) ‖ P): no evasion of ordinary PAC-Bayes.
    - Tests (all pass): brute-force identity E[e^{−C} g(w_K)] = E_P g on a tiny net (2·10⁶ runs; AIS and SMC); negative
      controls (Metropolis test removed; block scored after training on it) are caught decisively; 120 coverage runs at
      n = 1000 with no violation; random labels never certified below 0.9 (min 0.949 at n = 1000; 0.924 at 60k).
    - Floating point: cannot be "bounded" (a non-bijective step makes the path ratio infinite, not slightly off), so
      removed: `experiments/exact_hmc.py`, lattice HMC with integer-shear leapfrog (exact involution), exact discrete
      Gaussians (Canonne–Kamath–Steinke) for prior and momenta, exact rational energies, exact Bernoulli acceptance;
      determinism audited. Exact identity test: E[e^{−C}] = 1.003 ± 0.007 (5·10⁵ chains, 30,090 round trips, 0
      failures). Exact 60k runs: MNIST 0.0732 / 0.0716 (two seeds), Fashion 0.2206; same wall time as float.
    - Literature (search): closest is Maurer, Mirzaei & Pontil (arXiv:2510.06028), temperature-integral bounds for the
      exact Gibbs posterior, with plug-in estimates and a per-setting calibration, on binary MNIST at n = 8k; Harel et
      al. 2025 (second law). Bit-reversible integrators are known (Levesque & Verlet 1993); using them for exact
      certificates looks new. Floating-point attacks on DP-SGD are documented (Jin et al. 2022), so the exactness point
      also applies to our §14 certificates.
    - CNN (pixels, 29k params, GPU float64): n = 5000 pilot not better than Gabor + MLP; acceptance low, step tuning
      running; 60k ≈ 9 GPU-hours.
    - Infrastructure: `ais_cert.py` refactored (one `sampler` for runs and tests; `--arch cnn`, `--random_labels`,
      `--exact`); a wifi drop removed the web tools mid-session; they came back with the connection.

## Next steps (as of 2026-11-21, after entry 36)
- AIS certificate (§17): (a) an independent read of the proof of Theorem 2 by the author or a colleague; (b) exact
  lattice rerun of the scattering run (0.0648) and of every reported headline number; (c) data vs temperature
  annealing head to head (§17.9); (d) CNN after step tuning; CIFAR-10 with scattering features; (e) exact (discrete)
  noise for the DP-SGD certificates (§14), since float Gaussian noise does not satisfy the RDP premise exactly;
  (f) bridge to SGD: an SGD-trained network as a proposal inside a reversible kernel (allowed by Theorem 2).

## Next steps (as of 2026-11-20, updated after entry 32)
0. DP/MI route: (a) DONE via L-CL-Z Thm 8 (entry 27); structured multi-epoch tail examined in entry 28: worth
   ≤ 0.008 at measured gradient diversity, dropped. (a′) individual (Rényi-filter) accounting: tried, no gain (entry 29); (b) CIFAR-10 with data-free features (ScatterNet) against
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
