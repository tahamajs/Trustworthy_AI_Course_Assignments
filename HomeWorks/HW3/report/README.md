Assignment LaTeX template

Files created
- `assignment_template.tex` — main template (edit metadata at top: \authorname, \studentid, \assignment, ...)
- `references.bib` — sample bibliography entries
- `Makefile` — `make pdf` builds the PDF

How to use
1. Put figures exported from your notebooks in `report/figures/` (create the folder).
2. Edit document metadata near the top of `assignment_template.tex`:
   - `\authorname{}`, `\studentid{}`, `\assignment{}`
3. Add citations to `references.bib` and cite with `\citep{key}`.
4. Insert code snippets with `\begin{lstlisting}...\end{lstlisting}` or include a whole file with `\lstinputlisting{path/to/file.py}`.
5. Build: `make pdf` (requires `pdflatex` + `bibtex` or `latexmk`)

Build report
- **Build PDF only** (requires figures already generated): `make pdf` from `report/`.
- **Full build** (generates figures then PDF): `./build_report.sh` from HW3 root. This runs `generate_report_artifacts.py` then `make pdf`.
- **Prerequisites**: Python venv with `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`; models and `.npy` results in `code/q5_codes/`.

Notes & tips
- For syntax-highlighting using Pygments, consider replacing `listings` with `minted` (requires `-shell-escape`).
- If you prefer XeLaTeX for special fonts: change documentclass options and compile with `xelatex`.
- Keep images in `figures/` and refer to them in the LaTeX file (example provided).

Example workflow for notebook -> report
- Export important plots from Jupyter (`.png`/`.pdf`) into `report/figures/`.
- Copy key code blocks into the Appendix or `\lstinputlisting` the script files under `HW*/code/`.
- Update `references.bib` with any papers or libraries you cite.
