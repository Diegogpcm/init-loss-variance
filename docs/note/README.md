# Review note: prequential AIS certificate

`ais_certificate.tex` is a self-contained, paper-style write-up of the certificate developed in `docs/theory.md` §17:

1. Motivation and intuition, and how the method relates to existing certificates (Table 1).
2. Background definitions: PAC-Bayes and its disintegrated form, Gibbs posteriors and the prequential decomposition of the
   marginal likelihood, Markov kernels and reversibility, HMC, annealed importance sampling.
3. The method: algorithm, path identity (Lemma 1), certificate (Theorem 2), what the complexity measures
   (Propositions 3 and 4), other annealing paths, selection, particles.
4. Why floating point breaks the hypotheses, and the exact lattice implementation.
5. Experiments: validity tests and results, with comparisons.
6. Related work, limitations; appendices with the particle and lattice proofs and a reviewer checklist.

Build (standard TeX Live packages only, including TikZ/pgfplots; no .bib file):

    tectonic ais_certificate.tex                                  # locally (Tectonic installed via Homebrew)
    pdflatex ais_certificate.tex && pdflatex ais_certificate.tex  # or with TeX Live

or upload the single .tex file to Overleaf.

`docs/theory.md` §17 remains the working notebook; this note is the clean version and should be kept in sync with it.
