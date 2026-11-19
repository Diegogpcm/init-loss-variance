# Literature notes

What each work actually does, read in full unless marked *(abstract/snippet only)*. Last updated 2026-11-19.

## Closest to this project

**Picard-Weibel, Clerico, Moscoviz, Guedj (2025). How good is PAC-Bayes at explaining generalisation? arXiv:2503.08231.**
Purely theoretical; no new bound, no experiments of their own.
- Thm 1: for bounds of the form b(Q[R], P[D(dQ/dP)]) (Catoni, McAllester, Maurer, Bégin et al.; excludes Bernstein /
  coin-betting bounds), inf_Q B depends only on the push-forward of the prior by the empirical risk ("risk prior").
  Remark: if random labels give the same risk prior, the bound cannot distinguish them.
- Thm 2: the optimum is monotone in the (first-order) stochastic order. Distributions with equal means are all
  incomparable, so the theorem says nothing about variance at fixed mean (gap filled by our convex-order remark,
  theory.md §3).
- Cor 1–2, Thm 3 (Catoni): to certify G the prior needs mass ≳ α·exp(−2G²n − ln δ) on risk ≤ G/(1−α). Example: 1.5% on
  60k points (δ = 0.035) requires > 5.3e-12 mass below 1.65% error.
- App. B, Table 1: re-analysis of Pérez-Ortiz et al.; test-set bounds on the prior beat the PAC-Bayes certificate in
  13 of 14 cases. App. D: toy model (random labels per cluster) gives 1e-20 to 1e-378 mass at <1.5% error.
- Thesis: PAC-Bayes *witnesses* rather than explains generalization unless further structure is built in.

**Valle-Pérez, Camargo, Louis (2019, ICLR). Deep learning generalizes because the parameter-function map is biased
towards simple functions. arXiv:1805.08522.**
No new bound: applies Langford & Seeger's realizable PAC-Bayes bound (Cor. 1: −ln(1−ε) ≤ (ln 1/P(U) + ln(2m/δ))/(m−1))
with a *function-space* prior induced by random parameters. P(U) = prior mass of zero-training-error functions
(marginal likelihood), computed via NNGP + expectation propagation. Binarized MNIST / Fashion / CIFAR-10, m = 10k,
CNN and 1-hidden-layer FC; label-corruption sweep. Table 1 bounds: MNIST 0.134 / 0.169, Fashion 0.175 / 0.188,
CIFAR 0.485 / 0.518 (CNN / FC). Assumption: SGD samples the zero-error region ~uniformly. App.: bound best near the
order/chaos transition in weight variance (cf. our gain sweep).

**Valle-Pérez, Louis (2020). Generalization bounds for deep learning. arXiv:2012.04115.**
Review plus desiderata D.1–D.7 for generalization predictors. Thm 5.1: minor variant of the realizable bound holding
w.h.p. over posterior samples (authors: "essentially the same as Langford and Seeger"). Thm 6.4 (new): asymptotically
optimal up to a constant when learning curves are power laws. Kernel estimated by MC over M random initializations,
then EP. 19 architectures × 5 datasets × several m; bound tracks learning curves and correlates with error across
architectures, but differences between architectures are small. D.4 (optimizer dependence) explicitly out of reach.
Lists NAS as future work.

**Park, Lee, Peng, Cao, Sohl-Dickstein (2020). Towards NNGP-guided neural architecture search. arXiv:2011.06006.**
*(abstract only)* NNGP performance (kernel by MC over finite nets at init) as a cheap NAS signal on NAS-Bench-101 and
MNAS. No PAC-Bayes.

**Gracheva (2021). Trainless model performance estimation ... (CVU). Array 12. arXiv:2103.08312.** CV of untrained
accuracy over inits; lower CV better on NAS-Bench-201. In our MLP settings *higher* CVU is better (sign flip, unexplained).
CVU = std/mean of untrained accuracy = √(0-1 between-init variance)/(1/C), i.e. the 0-1 analogue of our proxy.

**Bowden, Csaposs, Hoffmann (2021). PAC-Bayes inspired NAS without training. Caltech CS159 report.** Motivated by
PAC-Bayes "evidence" (volume of solutions); untrained accuracy averaged over inits was *not* predictive (20 archs);
final objectives: contrastive triplet loss and MC-dropout perturbation, averaged over 100 inits. MNIST CNN blocks.

## Deep-learning PAC-Bayes certificates

**Dziugaite & Roy (2017). Computing nonvacuous generalization bounds for deep (stochastic) neural networks ...
arXiv:1703.11008.** Gaussian prior centred at the random init (variance optimized via union bound), Gaussian posterior
around SGD solution. Binary MNIST, n = 55k: KL 5,144–8,558 nats, bounds 0.161–0.223; random labels KL 201,131.
SNN test error 0.034 vs SGD 0.018 (the stochastic-predictor tax).

**Dziugaite & Roy (2018). Data-dependent PAC-Bayes priors via differential privacy.** Also evaluates Lever et al.'s
localized bound with SGLD (reported in Pérez-Ortiz Table 2): 0.26 at 12% error, vacuous at 6%.

**Pérez-Ortiz, Rivasplata, Shawe-Taylor, Szepesvári (2021). Tighter risk certificates for neural networks. JMLR.
arXiv:2007.12911.** Objectives f_quad, f_lambda, f_classic, f_bbb; priors: Gaussian at random init, or learned on
half the data. PAC-Bayes-kl with Langford–Caruana MC correction. MNIST Table 1: random-init CNN f_quad cert 0.2165,
KL/n 0.1039; learned-prior CNN cert 0.0155, KL/n 0.0001 (prior's own test error 1.04%). KL-attenuation trick.
**No localization**: Lever et al. mentioned only as an alternative.

**García-Pérez, Parrado-Hernández, Shawe-Taylor (2025). Some theoretical improvements on the tightness of PAC-Bayes
risk certificates for neural networks. arXiv:2510.07935.** New closed-form relaxations of Maurer's bound, a training
objective derived from it, KL-modulation (η possibly > 1), method to optimize bounds on non-differentiable losses.
Data-free priors only (data-dependent priors excluded on purpose); first non-vacuous CIFAR-10 PAC-Bayes certificates
for NNs (4–5 layer CNNs). 150,000 MC samples. Observes the bound's variance is much smaller than the prior loss's
variance ("good minima spread throughout parameter space; KL restricts the search"). **No localization.**

**Lotfi, Finzi, Kapoor, Potapczynski, Goldblum, Wilson (2022). PAC-Bayes compression bounds so tight that they can
explain generalization. arXiv:2211.13609.** Data-independent prior over compressed descriptions (random subspace
training + quantization + arithmetic coding). MNIST 11.6% bound with ~0.38 KB model; first non-vacuous data-free bounds
on FashionMNIST / CIFAR-10 / CIFAR-100.

**Haddouche & Guedj (2022). Online PAC-Bayes learning. NeurIPS. arXiv:2206.00024.** Thm 2.3: PAC-Bayes bound with a
sequence of priors P_i that may depend on the past (z_<i) and posteriors Q_i that may depend on everything, charged
Σ KL(Q_i‖P_i)/λ; certifies Σ E_{Q_i}[E(ℓ(h, z_i) | F_{i−1})] (no i.i.d. assumption needed). Corollaries: OPBTrain
(Gibbs updates Q̂_{i+1} ∝ e^{−λℓ(·, z_i)}Q̂_i; analogy with online gradient descent, λ ↔ step size) and OPBTest (Q = P,
KL-free; progressive validation). Disintegrated variants (Sec. 4). Small experiments on online regression/classification.
Hoeffding-type slack λmK²/2.

## Theory used

- Catoni (2007) monograph, *PAC-Bayesian supervised classification: the thermodynamics of statistical learning*:
  localized priors (theory.md §6).
- Lever, Laviolette, Shawe-Taylor (2013, TCS). Distribution-dependent priors e^{−γL}P₀.
- Germain, Bach, Lacoste, Lacoste-Julien (2016, NeurIPS). PAC-Bayes with NLL ⇔ marginal likelihood.
- Langford & Seeger (2001); Maurer (2004); McAllester (1998, 1999): PAC-Bayes-kl.
- Hoeffding (1963): Chernoff/kl bound for independent, non-identically distributed [0,1] variables (block estimator).
- Xu & Raginsky (2017) MI bounds; Steinke & Zakynthinou (2020) CMI; Livni & Moran (2020) *A limitation of the
  PAC-Bayes framework* (thresholds).
- Mingard, Valle-Pérez, Skalse, Louis (2021, JMLR). SGD's function distribution ≈ Bayesian posterior.
- Mandt, Hoffman, Blei (2017, JMLR). SGD as approximate Bayesian inference (Lyapunov stationary covariance).
- Maddox et al. (2019). SWAG: Gaussian from SGD iterates (diag + low rank).
- Mou et al. (2018); Negrea et al. (2019); Haghifam et al. (2020); Neu, Dziugaite, Haghifam, Roy (2021): trajectory /
  information-theoretic bounds for SGLD and SGD.
- Dziugaite, Hsu, Gharbieh, Arpino, Roy (2021). On the role of data in PAC-Bayes bounds (data-dependent prior ≈ ideal
  distribution-dependent prior).
- Normalization and scale invariance: van Laarhoven (2017); Hoffer et al. (2018); Arora, Li, Lyu (2019); Salimans et
  al. (2016, virtual BN). Reparameterization critique: Dinh et al. (2017). Magnitude-aware PAC-Bayes measures:
  Jiang et al. (2020), *Fantastic generalization measures*.
- Annealed IS / bidirectional MC: Grosse et al. (2015). Nested sampling: Skilling (2006).

## Prior mass implied by published certificates (derived here)

| Source | Prior | Task, n | Region | −ln mass (nats) | Mass | Nats / example | Certificate |
|---|---|---|---|---|---|---|---|
| Dziugaite & Roy 2017 | Gaussian at init | binary MNIST, 55k | near SGD solution, ~3% train err | 5,144–8,558 | 1e-2234 – 1e-3717 | 0.09–0.16 | 0.16–0.22 |
| Pérez-Ortiz 2021 | Gaussian at init | MNIST, 60k | ~5% train err | ≈6,200 (CNN), 8,300 (FCN) | ≈1e-2700, 1e-3600 | 0.10–0.14 | 0.22 / 0.32 |
| Pérez-Ortiz 2021 | learned on half the data | MNIST | ~1% train err | ≈2–5 | ≈1e-1 – 1e-2 | 0.0001 | 0.016 |
| Valle-Pérez 2019 | function space, NNGP | binary MNIST / Fashion / CIFAR, 10k | all zero-error functions | ≈1,430 / 1,910 / 6,620 | ≈1e-620 / 1e-830 / 1e-2880 | 0.14 / 0.19 / 0.66 | 0.13 / 0.18 / 0.49 |
| Lotfi 2022 | compression prior | MNIST, 60k | one ~380-byte model | ≈2,150 (≈3,100 bits) | ≈1e-930 | 0.036 | 0.116 |

Method: KL as published (Dziugaite & Roy), KL/n × n (Pérez-Ortiz), inverted realizable bound with δ = 0.05 assumed
(Valle-Pérez), model size in bits (Lotfi). These are lower estimates of the prior mass (posteriors not optimal).
