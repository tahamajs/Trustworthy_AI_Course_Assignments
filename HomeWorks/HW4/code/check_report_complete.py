#!/usr/bin/env python3
"""
HW4 Report validation script.
Runs the full pipeline, checks artifacts, compiles the report, and runs tests.
Reports status for Security, Privacy, Fairness, Report PDF, and Tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
REPORT_DIR = ROOT_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"
RESULTS_DIR = REPORT_DIR / "results"

EXPECTED_FIGURES = [
    "trigger_reconstructed.png",
    "trigger_all_labels_grid.png",
    "security_scale_profile.png",
    "security_before_after.png",
    "security_confusion_before_after.png",
    "security_unlearning_sweep.png",
    "privacy_scenarios.png",
    "privacy_tail_curves.png",
    "privacy_epsilon_sweep.png",
    "fairness_comparison.png",
    "fairness_group_rates.png",
    "fairness_tradeoff.png",
    "fairness_swapk_sweep.png",
]

REQUIRED_MACROS = [
    "QOneDetectedLabel",
    "QTwoBBase",
    "QThreeBaseAcc",
]


def check_prereqs() -> dict[str, bool | str]:
    """Check pre-requisites before running pipeline."""
    status = {}
    data_csv = CODE_DIR / "data.csv"
    status["data.csv"] = data_csv.exists()
    if not status["data.csv"]:
        status["data.csv_note"] = f"Missing: {data_csv}"

    model_dir = CODE_DIR / "model_weights" / "poisened_models"
    archive = CODE_DIR / "poisened_models.rar"
    has_models = any((model_dir / f"poisened_model_{i}.pth").exists() for i in range(10))
    status["poisoned_models"] = has_models or archive.exists()
    if not status["poisoned_models"]:
        status["poisoned_models_note"] = (
            "Place poisened_models.rar in code/ or extract poisened_model_0.pth ... poisened_model_9.pth "
            "into model_weights/poisened_models/"
        )
    return status


def run_pipeline(student_id: str = "810101504") -> bool:
    """Run generate_report_figs.py. Returns True if exit code 0."""
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / "generate_report_figs.py"), "--student-id", student_id],
        cwd=str(CODE_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout)
    return result.returncode == 0


def check_figures() -> tuple[list[str], list[str]]:
    """Return (present, missing) figure names."""
    present = []
    missing = []
    for name in EXPECTED_FIGURES:
        if (FIGURES_DIR / name).exists():
            present.append(name)
        else:
            missing.append(name)
    return present, missing


def check_macros() -> tuple[bool, str]:
    """Check results_macros.tex exists and has required macros."""
    macro_path = RESULTS_DIR / "results_macros.tex"
    if not macro_path.exists():
        return False, "results_macros.tex not found"
    text = macro_path.read_text(encoding="utf-8")
    for macro in REQUIRED_MACROS:
        if f"\\{macro}" not in text:
            return False, f"Macro {macro} not defined"
    return True, "OK"


def check_metrics_json() -> tuple[bool, str]:
    """Check metrics_summary.json exists and has expected keys."""
    path = RESULTS_DIR / "metrics_summary.json"
    if not path.exists():
        return False, "metrics_summary.json not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    for key in ("security", "privacy", "fairness"):
        if key not in data:
            return False, f"Missing key: {key}"
    return True, "OK"


def compile_report() -> bool:
    """Compile report PDF. Returns True if successful."""
    result = subprocess.run(
        ["make", "pdf"],
        cwd=str(REPORT_DIR),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and (REPORT_DIR / "assignment_template.pdf").exists()


def run_tests() -> bool:
    """Run pytest. Returns True if all pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(CODE_DIR),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> int:
    student_id = os.environ.get("HW4_STUDENT_ID", "810101504")
    print("=" * 60)
    print("HW4 Report Validation")
    print("=" * 60)

    # Pre-requisites
    prereqs = check_prereqs()
    print("\n[Pre-requisites]")
    for k, v in prereqs.items():
        if k.endswith("_note"):
            continue
        status = "OK" if v else "MISSING"
        print(f"  {k}: {status}")
        if not v and f"{k}_note" in prereqs:
            print(f"    -> {prereqs[f'{k}_note']}")

    # Run pipeline
    print("\n[Pipeline] Running generate_report_figs.py ...")
    pipeline_ok = run_pipeline(student_id)
    print(f"  Pipeline exit: {'OK' if pipeline_ok else 'FAILED'}")

    # Post-run checks
    present, missing = check_figures()
    print(f"\n[Figures] {len(present)}/{len(EXPECTED_FIGURES)} present")
    if missing:
        for m in missing[:6]:
            print(f"  Missing: {m}")
        if len(missing) > 6:
            print(f"  ... and {len(missing) - 6} more")

    macros_ok, macros_msg = check_macros()
    print(f"\n[Macros] {macros_msg}")

    metrics_ok, metrics_msg = check_metrics_json()
    print(f"[Metrics JSON] {metrics_msg}")

    # Compile report
    print("\n[Report] Compiling PDF ...")
    pdf_ok = compile_report()
    print(f"  PDF build: {'OK' if pdf_ok else 'FAILED'}")

    # Tests
    print("\n[Tests] Running pytest ...")
    tests_ok = run_tests()
    print(f"  Tests: {'PASS' if tests_ok else 'FAIL'}")

    # Summary table
    print("\n" + "=" * 60)
    print("Status Summary")
    print("=" * 60)
    security_ok = len([f for f in present if "security" in f or "trigger" in f]) >= 6
    privacy_ok = len([f for f in present if "privacy" in f]) >= 3
    fairness_ok = len([f for f in present if "fairness" in f]) >= 4
    print(f"  Security (figures): {'OK' if security_ok else 'Partial/Missing'}")
    print(f"  Privacy (figures):  {'OK' if privacy_ok else 'Partial/Missing'}")
    print(f"  Fairness (figures): {'OK' if fairness_ok else 'Partial/Missing'}")
    print(f"  Report PDF:         {'OK' if pdf_ok else 'FAILED'}")
    print(f"  Tests:              {'PASS' if tests_ok else 'FAIL'}")
    print("=" * 60)

    if not prereqs.get("poisoned_models", False):
        print("\n[Note] Without poisoned models, security figures will be placeholders.")
        print("       Privacy and fairness still run. Place poisened_models.rar in code/ to fix.")

    all_ok = pipeline_ok and macros_ok and metrics_ok and pdf_ok and tests_ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
