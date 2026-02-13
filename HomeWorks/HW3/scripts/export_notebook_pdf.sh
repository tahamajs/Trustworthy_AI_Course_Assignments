#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_PATH="${1:-$ROOT_DIR/code/HW3_complete_assignment.ipynb}"
OUT_DIR="${2:-$ROOT_DIR/output/pdf}"
OUT_NAME="${3:-hw3_complete_assignment}"
VENV_ACTIVATE="/Users/tahamajs/Documents/uni/venv/bin/activate"

if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

mkdir -p "$OUT_DIR"

echo "[1/4] Converting notebook to LaTeX..."
jupyter nbconvert \
  --to latex \
  "$NOTEBOOK_PATH" \
  --output "$OUT_NAME" \
  --output-dir "$OUT_DIR" \
  --TemplateExporter.exclude_input_prompt=True \
  --TemplateExporter.exclude_output_prompt=True

TEX_FILE="$OUT_DIR/$OUT_NAME.tex"
LOG_FILE="$OUT_DIR/$OUT_NAME.xelatex.log"
PDF_FILE="$OUT_DIR/$OUT_NAME.pdf"

echo "[2/4] Patching LaTeX for robust Unicode/Persian rendering..."
python3 - "$TEX_FILE" <<'PY'
from pathlib import Path
import sys

tex_path = Path(sys.argv[1])
text = tex_path.read_text(encoding="utf-8")

# Keep rendered text/bookmarks stable with common Unicode fallbacks.
text = text.replace("∣", "|")
text = text.replace("𝜖", r"\(\epsilon\)")
text = text.replace("𝛿", r"\(\delta\)")

# Safety fallback for the known markdown->LaTeX conversion edge case.
text = text.replace(
    r"classifier: (h=\operatorname{sgn}(X\_1 + 5X\_2 - 225000))",
    r"classifier: $h=\mathrm{sgn}(X_1 + 5X_2 - 225000)$",
)

font_anchor = r"\usepackage{unicode-math}"
font_patch = "\n".join(
    [
        r"\usepackage{unicode-math}",
        r"\setmainfont{Arial Unicode MS}",
        r"\setsansfont{Arial}",
        r"\setmonofont{Arial Unicode MS}",
    ]
)
if r"\setmainfont{" not in text and font_anchor in text:
    text = text.replace(font_anchor, font_patch, 1)

tex_path.write_text(text, encoding="utf-8")
PY

echo "[3/4] Building PDF with XeLaTeX (2 passes)..."
: > "$LOG_FILE"
for _ in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory "$OUT_DIR" "$TEX_FILE" >> "$LOG_FILE" 2>&1
done

echo "[4/4] Optional preview images..."
if command -v pdftoppm >/dev/null 2>&1; then
  PREVIEW_DIR="/tmp/hw3_notebook_pdf_preview"
  rm -rf "$PREVIEW_DIR"
  mkdir -p "$PREVIEW_DIR"
  pdftoppm -png "$PDF_FILE" "$PREVIEW_DIR/$OUT_NAME" >/dev/null 2>&1 || true
  echo "Preview images: $PREVIEW_DIR"
else
  echo "pdftoppm not found; skipping PNG preview rendering."
fi

echo "PDF generated: $PDF_FILE"
echo "XeLaTeX log:  $LOG_FILE"
