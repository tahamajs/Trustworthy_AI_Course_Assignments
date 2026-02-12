Assignment LaTeX template

Files created

- `assignment_template.tex` — main template (edit metadata at top: \authorname, \studentid, \assignment, ...)
- `references.bib` — sample bibliography entries
- `Makefile` — `make pdf` builds the PDF

How to use

1. Put figures exported from your notebooks in `report/figures/` (create the folder if missing).
2. Edit document metadata at the top of `assignment_template.tex`:
   - `\authorname{}`, `\studentid{}`, `\assignment{}`
3. Replace example numbers / placeholder figures in the `Results` section with your measured outputs and figures.
4. Add citations to `references.bib` and cite with `\citep{key}`.
5. Insert code snippets with `\begin{lstlisting}...\end{lstlisting}` or include a whole file with `\lstinputlisting{path/to/file.py}`.
6. Build PDF: `make pdf` (requires `pdflatex` + `bibtex` or `latexmk`).

Checklist before submission

- [ ] All figures exported to `report/figures/` and referenced in the tex file.
- [ ] Tables updated with final accuracy / loss numbers and standard deviations.
- [ ] Hyperparameters listed in the Appendix.
- [ ] All code referenced included (either in `code/` or via `\lstinputlisting`).

Notes & tips

- Use `latexmk -pdf` for a one-command build that runs bibtex automatically.
- For syntax highlighting with `minted`, add `-shell-escape` to your build command and replace `listings` with `minted`.
- Keep images at 300 DPI for publication-quality figures.

Example workflow for notebook -> report

- Export important plots from Jupyter (`.png`/`.pdf`) into `report/figures/`.
- Copy key code blocks into the Appendix or `\lstinputlisting` the script files under `HW1/code/`.
- Update `references.bib` with any papers or libraries you cite.

If you want, I can: (a) insert your actual experiment numbers into the template, or (b) export notebook plots into `report/figures/` for you. Reply with which option you prefer.
