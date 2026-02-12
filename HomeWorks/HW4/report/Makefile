PDFLATEX=pdflatex
BIBTEX=bibtex
TEXFILE=assignment_template.tex
OUTFILE=assignment_template.pdf

.PHONY: all pdf clean

all: pdf

pdf:
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
