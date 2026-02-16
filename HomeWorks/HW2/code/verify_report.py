"""Verify HW2 report pipeline: run generate_report_plots if needed, check figures, optionally build PDF.

Usage:
    python code/verify_report.py
    python code/verify_report.py --build-pdf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
REPORT_DIR = ROOT / "report"
CODE_DIR = ROOT / "code"

REQUIRED_FIGURES = [
    "class_distribution.png",
    "eda_correlation_matrix.png",
    "eda_pairplot.png",
    "eda_outlier_boxplots.png",
    "training_loss_curves.png",
    "confusion_matrix_comparison.png",
    "roc_pr_comparison.png",
    "calibration_comparison.png",
    "threshold_sensitivity.png",
    "permutation_importance_comparison.png",
    "lime_shap_agreement.png",
    "lime_shap_compare_sample_0.png",
    "lime_shap_compare_sample_1.png",
    "lime_shap_compare_sample_2.png",
    "shap_force_sample_0.png",
    "shap_force_sample_1.png",
    "shap_force_sample_2.png",
    "correlation_vs_shap_importance.png",
    "grace_counterfactual_shap_shift.png",
    "nam_feature_functions.png",
    "vgg16_six_image_predictions.png",
    "gradcam_demo.png",
    "gradcam_overlay_demo.png",
    "guided_gradcam_example.png",
    "smoothgrad_guided_comparison.png",
    "smoothgrad_guided_backprop.png",
    "smoothgrad_guided_gradcam.png",
    "smoothgrad_sample_sweep.png",
    "smoothgrad_convergence_metrics.png",
    "adversarial_fgsm_comparison.png",
    "feature_visualization_hen.png",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify HW2 report figures and optionally build PDF.")
    parser.add_argument("--build-pdf", action="store_true", help="Run make pdf in report/ after verifying figures.")
    args = parser.parse_args()

    missing = [f for f in REQUIRED_FIGURES if not (FIG_DIR / f).exists()]

    if missing:
        print(f"[verify] {len(missing)} figures missing. Running generate_report_plots.py...")
        sys.path.insert(0, str(CODE_DIR))
        import generate_report_plots
        generate_report_plots.main()
        missing = [f for f in REQUIRED_FIGURES if not (FIG_DIR / f).exists()]

    if missing:
        print(f"[verify] FAIL: {len(missing)} figures still missing after pipeline run:")
        for f in missing:
            print(f"  - {f}")
        return 1

    print(f"[verify] OK: all {len(REQUIRED_FIGURES)} required figures exist in {FIG_DIR}")

    if args.build_pdf:
        print("[verify] Building PDF...")
        result = subprocess.run(
            ["make", "pdf"],
            cwd=REPORT_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[verify] make pdf failed: {result.stderr}")
            return 1
        pdf_path = REPORT_DIR / "assignment_template.pdf"
        if pdf_path.exists():
            print(f"[verify] OK: PDF built at {pdf_path}")
        else:
            print("[verify] FAIL: PDF not found after make pdf")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
