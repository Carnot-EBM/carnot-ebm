Carnot Position Paper v3 -- arXiv submission bundle
===================================================

This directory contains the LaTeX source for the Carnot position
paper "An Architectural Framework for Mapping the Empirical Bounds
of LLM Verification" (draft v3, target submission 2026-05-15).

Source files
------------
- main.tex              Master LaTeX document (~7,500 words, 7 figures,
                        4 tables, structured per arXiv standards).
- carnot.bib            BibTeX bibliography (27 references).
- figures/fig1.pdf .. figures/fig7.pdf
                        Vector PDF figures generated from the
                        matplotlib scripts in docs/figures/. PDFs are
                        directly includable by pdflatex.

Source markdown the LaTeX was hand-converted from
-------------------------------------------------
- ../position-paper-draft-v3.md   (903 lines, the canonical v3 draft)

Note: pandoc was not available on the source machine; main.tex is a
manual conversion that preserves every section, equation, table, and
figure callout from the markdown source. Reviewers should treat
main.tex as the authoritative LaTeX representation; if they regenerate
from markdown via pandoc, the only expected diffs are around
math-mode escaping and figure-placement floats.

Build instructions
------------------
Standard LaTeX bibliography pipeline:

    cd docs/arxiv-paper/
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

The first pdflatex pass will produce undefined-citation warnings
which the bibtex + two trailing pdflatex passes resolve. Final
output: main.pdf (estimated 18-22 pages including figures and
appendices).

If pdflatex is not installed locally, install texlive-full
(Linux: `sudo pacman -S texlive-most texlive-latexextra` on Arch /
CachyOS; `sudo apt-get install texlive-full` on Debian/Ubuntu) or
upload the bundle to Overleaf (https://overleaf.com) and let it
build the PDF in the cloud.

arXiv upload pipeline
---------------------
1. Build PDF locally per the instructions above; verify it renders
   correctly and that all 7 figures appear.
2. Bundle main.tex, carnot.bib, and the figures/ directory into a
   single .tar.gz:
       tar czvf carnot-arxiv-v3.tar.gz main.tex carnot.bib figures/
3. Upload to https://arxiv.org/submit, primary category cs.LG,
   secondary categories cs.AI, quant-ph, cs.NE.
4. arXiv will run pdflatex + bibtex on its own. Expect a 6-12 hour
   moderation queue before the preprint is publicly visible.

Validation status
-----------------
- pdflatex: NOT INSTALLED on the build machine (2026-05-01).
  Bundle is therefore "ready for pdflatex elsewhere" (Overleaf,
  arXiv submission server, or a fresh texlive install).
- All figure PDFs are present and readable.
- Bibliography keys cross-checked against \cite{} commands in
  main.tex.

Author signature
----------------
Author placeholder in main.tex must be filled in before submission.
The conductor agent does not commit author identity; that is
operator policy.
