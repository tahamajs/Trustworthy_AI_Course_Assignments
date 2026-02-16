PDFLATEX=pdflatex
BIBTEX=bibtex
TEXFILE=assignment_template.tex
OUTFILE=assignment_template.pdf
CODE_DIR=../code
STUDENT_ID?=810101504

.PHONY: all pdf report clean

all: pdf

# Full report: generate figures/macros first, then compile PDF.
# Macros (results/results_macros.tex) must exist before LaTeX runs.
report:
	@echo "Generating figures and macros..."
	cd $(CODE_DIR) && python generate_report_figs.py --student-id $(STUDENT_ID)
	@$(MAKE) pdf

pdf:
	@mkdir -p figures
	@echo "Compiling $(TEXFILE) -> $(OUTFILE)"
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -pdflatex="$(PDFLATEX) -interaction=nonstopmode" $(TEXFILE); \
	else \
		$(PDFLATEX) -interaction=nonstopmode $(TEXFILE) || true; \
		$(BIBTEX) $(basename $(TEXFILE)) || true; \
		$(PDFLATEX) -interaction=nonstopmode $(TEXFILE) || true; \
		$(PDFLATEX) -interaction=nonstopmode $(TEXFILE) || true; \
	fi

clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.fls *.fdb_latexmk *.pdf
