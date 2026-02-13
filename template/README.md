# LaTeX Template Guide

This folder provides reusable LaTeX templates for homework reports.

## Files

- `assignment_template.tex`: standard assignment/report layout.
- `paper_template.tex`: compact paper-style layout.
- `references.bib`: bibliography database.
- `Makefile`: build helpers (`make pdf`, `make paper`).

## Quick start

1. Edit metadata in the selected template (`\authorname`, `\studentid`, `\assignment`).
2. Export plots to `template/figures/`.
3. Add references to `references.bib` and cite via `\citep{key}`.
4. Build:
   - `make pdf` for `assignment_template.tex`
   - `make paper` for `paper_template.tex`

## Customization notes

- Accent color is defined in the TeX source (`\definecolor{accent}{...}`).
- Font package can be swapped by editing the corresponding `\usepackage{...}` line.
- `minted` can be used instead of `listings` if `-shell-escape` is enabled.
