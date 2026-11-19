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
17. **Documentation for a paper (author's request):** `docs/` created (theory, literature, experiments, this log,
    paper outline).

## Decisions and preferences
- Init-variance-as-NAS-proxy: deprioritized (2026-11-19).
- BN in PAC-Bayes: statistics as parameters preferred over reference sets (no held-out data).
- Data-dependent priors via sample splitting: avoided (author's paper excludes them; Picard-Weibel critique).
