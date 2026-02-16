#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/code/q5_codes"
python generate_report_artifacts.py
cd "$ROOT/report"
make pdf
echo "PDF: $ROOT/report/assignment_template.pdf"
