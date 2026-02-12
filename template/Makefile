PDFLATEX=pdflatex
BIBTEX=bibtex
TEXFILE=assignment_template.tex
OUTFILE=assignment_template.pdf

.PHONY: all pdf paper clean

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

paper:
	@echo "Compiling paper_template.tex -> paper_template.pdf"
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -pdflatex="$(PDFLATEX) -interaction=nonstopmode" paper_template.tex; \
	else \
		$(PDFLATEX) -interaction=nonstopmode paper_template.tex || true; \
		$(BIBTEX) paper_template || true; \
		$(PDFLATEX) -interaction=nonstopmode paper_template.tex || true; \
		$(PDFLATEX) -interaction=nonstopmode paper_template.tex || true; \
	fi

clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.fls *.fdb_latexmk *.pdf
