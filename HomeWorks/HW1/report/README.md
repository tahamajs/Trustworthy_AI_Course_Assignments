Assignment LaTeX template

Files created

- `assignment_template.tex` — main template (edit metadata at top: `\authorname`, `\studentid`, `\assignment`)
- `references.bib` — bibliography entries
- `Makefile` — `make pdf` builds the report

How to use

1. Put figures exported from notebooks in `report/figures/`.
2. Edit metadata near the top of `assignment_template.tex`.
3. Replace placeholder tables/figures with final experiment outputs.
4. Add references to `references.bib` and cite with `\citep{key}`.
5. Build the report: `make pdf` (requires `pdflatex` + `bibtex` or `latexmk`).

Checklist before submission

- [ ] All figures are in `report/figures/` and referenced.
- [ ] Result tables use final numbers.
- [ ] Hyperparameters are listed in an appendix.
- [ ] Cited works are included in `references.bib`.
