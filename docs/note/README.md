# Review note: prequential AIS certificate

`ais_certificate.tex` is a self-contained, paper-style write-up of the certificate in `docs/theory.md` §17: motivation and
assumptions, the path identity and the certificate with full proofs, the SMC extension, why floating point breaks the
AIS), the method with proofs, exactness, experiments, related work, limitations, appendices with the particle and lattice
proofs, and a reviewer checklist.

Build (standard TeX Live packages only; no .bib file). Locally: `tectonic ais_certificate.tex` (installed via Homebrew), or:

    pdflatex ais_certificate.tex && pdflatex ais_certificate.tex   # second pass fills the table of contents

or upload the single .tex file to Overleaf.

`docs/theory.md` §17 remains the working notebook; this note is the clean version and should be kept in sync with it.
