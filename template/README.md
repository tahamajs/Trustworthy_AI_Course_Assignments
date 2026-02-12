Beautiful assignment LaTeX template — usage notes

Files included
- `assignment_template.tex` — redesigned, modern template (edit metadata near the top)
- `references.bib` — sample bibliography entries
- `Makefile` — build with `make pdf`

Quick start
1. Edit metadata in `assignment_template.tex`: `\authorname{}`, `\studentid{}`, `\assignment{}`.
2. (Optional) Place exported figures in `template/figures/` — the template will show a tasteful placeholder if an image is missing.
3. Add citations to `references.bib` and use `\citep{key}`.
4. Insert code via `\begin{lstlisting}...\end{lstlisting}` or `\lstinputlisting{path/to/file.py}`.
5. Build: `make pdf` (needs `pdflatex` + `bibtex` or `latexmk`).

What changed in the visual update
- Improved typography (Palatino), colored accents, and refined section headings.
- Conditional figure placeholders (safe build even when example images are absent).
- Cleaner code listings and a boxed abstract for better readability.

Customization tips
- Change the accent color: edit `\definecolor{accent}{HTML}{2A9D8F}` in the .tex file.
- Swap font: change `\usepackage{mathpazo}` to another font package (e.g., `newpx` or `lmodern`).
- Use `minted` for advanced code highlighting (requires `-shell-escape` and Python Pygments).

Notes
- Export critical plots from your notebooks into `template/figures/` to replace placeholders.
- Keep `references.bib` updated with any papers or libraries you cite.

If you want, I can:
- switch to an IEEE / conference style, or
- enable `minted` and add build instructions for `-shell-escape`.
Tell me which and I’ll update the template.