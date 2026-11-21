# Literature notes

What each work actually does, read in full unless marked *(abstract/snippet only)*. Last updated 2026-11-20.

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
Numbers (verified 2026-11-20 from the HTML version): MNIST, MLP 3×600, δ = 0.025, 150k MC samples, σ₀ ∈ [0.02, 0.08]:
CE-trained f_rts 0.335 (Gibbs test 0.163); 0-1 objectives f̃_mb 0.325 ± 0.006; KL-modulated **0.315** (best).
CIFAR-10, 4-layer CNN (9,946 params) f̃_mb **0.738 ± 0.007**; 4-layer 315k params 0.758; 5-layer 0.756.

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

## Differential privacy / information-theoretic certificates

**Lampert, Cairney-Leeming & Zakerinia (2026, v2). From privacy to generalization: linear max-information bounds for
differentially private learning algorithms. arXiv:2605.26222v2.** *(PDF text read in full for Thms 7–8, App. B.3, App. C;
supersedes the v1 notes below.)*
- Thm 1: (ε,δ)-DP ⇒ I∞^β ≤ min{nε, ⅛nε² + ε√((n/2)ln(1/β₀)) + …}, with β₀ = β − nδ.
- Thm 2 = Thms 7–8, for (α, ε)-RDP (replace-one).
  - Thm 7 (Hölder over the n per-example increments): κ = n(ε + ln(1/β)/(α−1)).
  - **Thm 8**: κ = v⋆ + 2√(v⋆L₂) + L₂/(α−1), with v⋆ = n/(2(α−1)²)·asinh²√((e^{(α−1)ε} − 1)/β₁).
  - Proof of Thm 8: posterior-to-prior filtration over examples; symmetrised Rényi J_i with E J_i ≤ e^{(α−1)ε};
    backward integration under the product measure; Markov on (1/n)ΣJ_i inside a concave ψ.
  - **Poisson amplification enters through ε**: Opacus add/remove RDP is converted to replace-one by the weak triangle
    inequality (App. C), and α is chosen from 2…5000.
- Thm 3 (Gaussian mechanism) and Thm 4 (streaming DP-SGD) exploit Gaussian structure: κ ≈ I + O(√(I·L)) + O(L).
  Multi-epoch Poisson is left open for these structured bounds.
- PAC-Bayes Thm 5: kl ≤ (KL(ρ‖π) + I∞^{δ/2} + ln(4√n/δ))/n. Corollary 6 sets ρ = π, i.e. it certifies the DP-SGD output.
- Experiments use our pacbayesgradients code (García-Pérez et al.).
  - MNIST 3×600 MLP: data-free 0.325; streaming prior ρ 0.273; Poisson prior ρ **0.233**, h₀ 0.248 (κ = 1,224).
  - CIFAR-10: h₀ 0.780 / 0.797 / 0.848 (4 / 9 / 15 layers); ρ **0.716** (4 layers, against data-free 0.738), 0.782, 0.757.
- **Relation to us:** their Thm 8 is exactly our "open lever". Applying it to our recipe A (I_ub = 12) gives κ = 185
  (Markov: 522), so MNIST 0.084, Fashion 0.248, CIFAR-10 0.715 (theory.md §14.1). The gap to their h₀ numbers (0.248 /
  0.780) comes from the recipe: small models on data-free features with very noisy, short training (I ≈ 12 nats).

**[v1 notes] Lampert & Zakerinia (2026). From privacy to generalization: linear max-information bounds for DP-SGD.
arXiv:2605.26222v1.** *(read via HTML, appendix partly)* Thm 1: high-probability max-information of DP-SGD (streaming or
batch, disjoint batches, each example once per epoch, n ≥ Tm), I∞^β ≤ ½·E·T·ν + E·inf_λ (1/λ)[T·F((λ+λ²)/2) + ln(E/β)],
ν = mζ²/σ², F(x) = (16ν²x² + νx)/(1 − 2νx); leading term = n·E·ζ²/(2σ²) = nρ (linear in n, no Markov). Proof: Jensen
against an *independent copy* S̃ of the whole dataset, then per-step Gaussian log-likelihood expansion and an MGF bound.
Thm 2: PAC-Bayes-kl with a DP-SGD prior: kl ≤ (KL(ρ‖π) + κ + ln(4√n/δ))/n, κ = Thm 1 at β = δ/2. **No subsampling
amplification** (Poisson listed as future work, "worse constants"). Table 1: MNIST MLP, data-free prior 0.32 vs DP-SGD
prior 0.29 (κ = 42, KL 5,020, posterior train 0.12); CIFAR-10 ConvNet 0.91 vs 0.84. Closest work to our §14 (they bound
max-information and keep a Gaussian posterior; we bound MI with amplification and certify the network itself via
Markov).

**Tramèr & Boneh (2021). Differentially private learning needs better features (or much more data). ICLR.
arXiv:2011.11660.** *(from memory, not re-read)* Under DP, linear models on data-free ScatterNet features beat end-to-end
CNNs on MNIST / Fashion / CIFAR-10. Motivates our Gabor-feature runs (E-dp).

**Abadi et al. (2016)** DP-SGD (clipping + Gaussian noise, moments accountant). **Mironov (2017); Mironov, Talwar &
Zhang (2019)**: Rényi DP and the subsampled Gaussian. **Bun & Steinke (2016)** zCDP; I(X; M(X)) ≤ ρn for product
distributions. **Pensia, Jog & Loh (2018)**: MI bounds for noisy iterative algorithms (whole-dataset sensitivity, no 1/n).
**Rivasplata et al. (2020)**: PAC-Bayes with priors of small max-information; disintegrated bounds.

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

## Function-space / conditional (supersample) information bounds (read 2026-11-20, full text)

**Steinke & Zakynthinou (2020) setting.**
- A supersample Z̃ ∈ Z^{n×2}; S ∈ {0,1}^n selects the training half; the bounds compare the training loss with the
  loss on the *ghost* half.

**Harutyunyan, Raginsky, Ver Steeg & Galstyan (2021), NeurIPS. arXiv:2110.01584.**
- f-CMI = I(f(Z̃_S, X̃, R); S | Z̃): information in the *predictions* on all 2n inputs.
- Samplewise version (their Cor. 2): |E gen| ≤ (1/n) Σᵢ E √(2·I(f(z̃_S, x̃ᵢ, R); Sᵢ)). The information is between a
  pair of predictions and one bit, so it is easy to estimate.
- *Expected* bound only, with a square-gap version (Markov for tails).
- Estimated by retraining over many (z̃, S) draws. Close to the true gap for a CNN on MNIST 4 vs 9 and a ResNet-50
  fine-tuned on CIFAR-10 (bound about 3× the gap).
- **Fixed-seed (deterministic) training has f-CMI ≈ random-seed f-CMI** (their App. B, Fig. 3b), whereas weight-based
  I(W;S) is infinite or trivial.
- Remark 1: predictions on both halves are needed (test-only or train-only predictions fail).

**Hellström & Durisi (2022), NeurIPS. arXiv:2210.06422.**
- e-CMI (information in the *losses*), jointly-convex-function family, samplewise binary-kl bound.
- Thm 7 (high probability over Z̃, S): bounds the *ghost-half* loss by
  D(P_{λ|Z̃S} ‖ P_{λ|Z̃}) + log(√n/δ), with λ the full loss matrix.
- Notes that samplewise tail bounds are loose: linear in 1/δ.

**Hellström & Durisi (2020), IEEE JSAIT. arXiv:2005.08044.**
- Single-draw (disintegrated) bounds via the information density ı(W,Z), and its conditional version.
- Tails via Markov on ı, via central moments M_t of ı, or via Rényi / maximal leakage. Same structure as our §14.

**Wang & Mao (2023), ICML. arXiv:2302.02432.**
- Loss-difference CMI: project the loss pair to one number; Rademacher-correlated MI bounds; the tightest supersample
  bounds; exact in the interpolating regime.
- Expected bounds only.


## Sampler-based certificates and floating-point exactness (searched 2026-11-21; theory.md §17)

**Maurer, Mirzaei & Pontil (2025/26), "Generalization of Gibbs and Langevin Monte Carlo Algorithms in the
Interpolation Regime", arXiv:2510.06028 (v3, May 2026).** Closest prior work to §17.
- Disintegrated PAC-Bayes for one draw h from a data-dependent ν(x) (their Thm 2.1, eq. 2; bounded loss).
- Gibbs posterior G_β: ln dG_β/dπ(h) = ∫₀^β (E_{G_γ}[L̂] − L̂(h)) dγ (their Lemma 3.1): thermodynamic integration over
  *temperature*. Our §17 C is the realised, data-annealed counterpart (prequential sum over data blocks), exact for the
  actual run.
- LMC/ULA: covered in theory up to KL/TV/W2 convergence terms (LSI, Hessian bounds; Vempala–Wibisono discretisation),
  which are not computed in the experiments; E_{G_γ}[L̂] is a plug-in running mean along one chain (no concentration);
  reported bounds use a per-setting multiplier (0.08–0.89) calibrated so random-label bounds reach chance.
- Experiments: binary MNIST (0–4 vs 5–9), binary CIFAR-10, SVHN; n = 2000 or 8000; δ = 0.01; data-free Gaussian prior
  (σ = 5). MNIST 8k, β = 64k: test bound 0.0860 (2HL FC), 0.1314 (3HL), 0.0375 (LeNet-5) vs test error 0.0498 /
  0.0549 / 0.0317. Not certificates in our sense (uncertified estimates, calibration, mixing assumed); not comparable
  with our 10-class, full-data numbers.

**Harel, Wolanowsky, Vardi, Srebro & Soudry (2025), "Temperature is All You Need for Generalization in Langevin Dynamics
and other Markov Processes", arXiv:2505.19087.** Second-law argument: for a Markov process started at the prior with
Gibbs stationary law, KL(ν_t ‖ π) ≤ β E_π[L̂] along the whole trajectory; data-free, vacuous for β > n. Our C replaces
β E_π[L̂] (the prior's loss on all n points at once) by the prequential loss of the run.

**Viallard et al. (AISTATS 2024), "Leveraging PAC-Bayes Theory and Gibbs Distributions for Generalization Bounds with
Complexity Measures", arXiv:2402.13285.** Disintegrated bounds with Gibbs posteriors over arbitrary complexity
measures; related framing, no realised-path ratio.

**Bit-reversible integrators.** Levesque & Verlet (1993, J. Stat. Phys. 72:519) "Molecular dynamics and time
reversibility": positions stored as integers, forces rounded onto the grid, exactly reversible. Follow-ups: JANUS
(Rein & Tamayo 2017, arXiv:1704.07715); exact bitwise reversible integrator (NVIDIA 2022, arXiv:2207.07695). We found no
use of such integrators to make an MCMC kernel exactly reversible for a certificate (§17.5 looks new; search was
quick).

**Floating-point exactness in DP.** Mironov (2012, CCS) broke floating-point Laplace mechanisms. Jin, McMurtry,
Rubinstein & Ohrimenko, "Are We There Yet? Timing and Floating-Point Attacks on Differential Privacy Systems"
(arXiv:2112.05307, IEEE S&P 2022): the Gaussian samplers of NumPy, PyTorch and Go are vulnerable; up to 92.56% attack
success on DP-SGD in Opacus. Precision-based attacks and interval refining: arXiv:2207.13793. Exact discrete Gaussian:
Canonne, Kamath & Steinke (NeurIPS 2020). Relevance: our DP-SGD certificates (§14) sample float Gaussians, so their RDP
premise holds only for the idealised mechanism; the same applies to Gaussian PAC-Bayes posteriors sampled in float.

**Grosse, Ancha & Roy (NeurIPS 2016), "Measuring the reliability of MCMC inference with bidirectional Monte Carlo"
(BREAD), arXiv:1606.02275; Grosse, Ghahramani & Adams (2015), "Sandwiching the marginal likelihood using bidirectional
Monte Carlo", arXiv:1511.02543.** AIS/SMC treated as importance sampling on an extended space; forward and reverse
log-weights bound log-ML from both sides, and their gap upper-bounds (in expectation) the symmetrised KL between the
law of the AIS sample and the posterior (rigorous only on simulated data, which needs an exact posterior sample).
Closest relative of our "E[C] ≥ KL(law(w_K) ‖ P)" remark (§17.2): same extended-space log-weights, used there for
a KL to the posterior, here for a realised KL-type complexity to the data-free prior inside PAC-Bayes. Must be cited.

## Credit accounting for §17 (as of 2026-11-21)
- Borrowed: path identity (Neal 2001 AIS = discrete Jarzynski; Crooks; Del Moral, Doucet & Jasra 2006 backward
  kernels); particle Gibbs (Andrieu, Doucet & Holenstein 2010); disintegrated PAC-Bayes-kl (Blanchard & Fleuret 2007,
  Catoni 2007, Rivasplata et al. 2020; Viallard et al. 2024); Maurer (2004); extended-space log-weights as KL bounds
  (Grosse et al. 2015, 2016); annealing-integral bounds for Gibbs posteriors (Maurer, Mirzaei & Pontil 2025/26; Harel
  et al. 2025); prequential codes (Dawid; Blier & Ollivier 2018; Excess Description Length 2026); PAC-Bayes and
  marginal likelihood (Germain et al. 2016); bit-reversible integrators (Levesque & Verlet 1993); exact discrete
  Gaussian (Canonne, Kamath & Steinke 2020); floating-point attacks (Mironov 2012; Jin et al. 2022); scattering
  features (Mallat; Bruna & Mallat; Tramèr & Boneh 2021 for their use in certificates).
- Apparently new (quick searches only; a thorough search is required before any claim): (1) the realised
  extended-space log-ratio of an AIS/SMC run as the complexity in a disintegrated PAC-Bayes bound for its final sample,
  enabled by an extended prior that may depend on S because only its data-free w_K-marginal enters (no mixing
  assumption, no estimated expectations); (2) the data path (prequential charging, each point once) as that run;
  (3) bit-exact certificates via lattice HMC with exact discrete sampling; (4) the resulting 10-class, full-data
  data-free certificates (MNIST 0.0648, Fashion 0.2206).
