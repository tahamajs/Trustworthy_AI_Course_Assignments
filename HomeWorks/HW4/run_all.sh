#!/usr/bin/env bash
# HW4 run-all: validate tests, generate report artifacts, build PDF.
# Run from HomeWorks/HW4/ (project root).

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== Step 1: Run tests ==="
cd code
pytest tests -v
cd ..

echo ""
echo "=== Step 2: Generate report figures and macros ==="
cd code
python generate_report_figs.py --student-id 810101504 --security-profile fast-smoke
cd ..

echo ""
echo "=== Step 3: Build report PDF ==="
cd report
make pdf
cd ..

echo ""
echo "=== Step 4: Validate outputs ==="
REQUIRED_FIGS=(
  "report/figures/trigger_reconstructed.png"
  "report/figures/trigger_all_labels_grid.png"
  "report/figures/security_scale_profile.png"
  "report/figures/security_before_after.png"
  "report/figures/security_confusion_before_after.png"
  "report/figures/security_unlearning_sweep.png"
  "report/figures/privacy_scenarios.png"
  "report/figures/privacy_tail_curves.png"
  "report/figures/privacy_epsilon_sweep.png"
  "report/figures/fairness_comparison.png"
  "report/figures/fairness_group_rates.png"
  "report/figures/fairness_tradeoff.png"
  "report/figures/fairness_swapk_sweep.png"
)
MISSING=0
for f in "${REQUIRED_FIGS[@]}"; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "Missing: $f"
    MISSING=$((MISSING + 1))
  fi
done
if [[ $MISSING -gt 0 ]]; then
  echo "WARNING: $MISSING figure(s) missing. Run generate_report_figs.py with full assets (poisoned models, MNIST) to produce all figures."
fi

if [[ ! -f "$ROOT/report/assignment_template.pdf" ]]; then
  echo "ERROR: PDF not produced."
  exit 1
fi

echo ""
echo "=== Done ==="
echo "Report: $ROOT/report/assignment_template.pdf"
