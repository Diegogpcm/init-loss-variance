# Review note: prequential AIS certificate

`ais_certificate.tex` is a self-contained write-up of the certificate in `docs/theory.md` §17: setting and
assumptions, the path identity and the certificate with full proofs, the SMC extension, why floating point breaks the
hypotheses, the exact lattice implementation, validity tests, results, related work and credit, and a checklist of the
steps to scrutinise.

Build (standard TeX Live packages only; no .bib file):

    pdflatex ais_certificate.tex && pdflatex ais_certificate.tex   # second pass fills the table of contents

or upload the single .tex file to Overleaf.

`docs/theory.md` §17 remains the working notebook; this note is the clean version and should be kept in sync with it.
