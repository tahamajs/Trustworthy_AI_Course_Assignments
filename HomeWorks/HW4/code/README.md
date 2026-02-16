# HW4/code — Implementation & usage details

This document describes how to run the security, privacy, and fairness code, where test assets live, and how to reproduce the report figures.

---

## Key scripts & what they do

- `neural_cleanse.py` — trigger optimization + MAD-based target detection + unlearning demo.
- `privacy.py` — Laplace mechanism helpers and composition helpers used for assignment questions.
- `fairness.py` — fairness metrics and assignment-compliant promotion/demotion mitigation (CP/CD by true labels).
- `generate_report_figs.py` — builds PNG figures and LaTeX macros used in the report.
- `tests/` — unit tests for each sub-module (run with `pytest`).

---

## Poisoned model setup (Q1 Security)

The model index is chosen from the **last digit of your student ID** (e.g. `810101504` → last digit **4** → `poisened_model_4.pth`).

- **Option A:** Put `poisened_models.rar` in this directory (or pass `--archive-path`). Install `unar` (The Unarchiver). The script will extract into `model_weights/poisened_models/`.
- **Option B:** Extract the archive manually so `model_weights/poisened_models/` contains `poisened_model_0.pth` … `poisened_model_9.pth`.

---

## Quick examples

- Run unit tests:

  ```bash
  pytest tests
  ```

- Generate all report figures and metrics (use your student ID so the correct Q1 checkpoint is used):

  ```bash
  python generate_report_figs.py --student-id YOUR_STUDENT_ID
  ```

  Or set the model index explicitly: `--model-index 0` … `--model-index 9`. With MNIST offline: `--no-download-mnist` (data read from `data/mnist/` or `--mnist-root`).

- Run Neural Cleanse pipeline on a supplied model:

  ```py
  from neural_cleanse import reconstruct_trigger, detect_outlier_scales
  mask, pattern, scale = reconstruct_trigger(model, data_loader, target_label=3)
  outlier_label = detect_outlier_scales(scales)
  ```

---

## Model weights & poisoned examples

- `model_weights/poisened_models/` must contain the 10 poisoned checkpoints (see “Poisoned model setup” above). Used by the security pipeline and unit tests.

- For Q2 (Privacy), the same numbers used in the report are computed by `generate_report_figs.py` and written to `report/results/metrics_summary.json`. Running `python privacy.py` prints a short copy-paste summary of income and counting query results. For a report-ready summary of all Q2 answers, run `python print_privacy_answers.py` from this directory.

- For Q3 (Fairness), the assignment fair model uses **true labels** to define cohorts: CP = men with income >50k, CD = women with income ≤50k; ranking is by model probability. The swap count `k` is set via `--swap-k` (default 10). Using `n = min(|CP|, |CD|)` as `k` is a reasonable choice to balance swaps across groups.

---

## Testing & validation

- The test suite verifies basic numerical properties and shapes and serves as a quick smoke-check for pipeline changes.
- Add tests in `tests/` when adding functionality; keep tests deterministic (fix seeds where randomness is used).

---

## Notes & limitations

- `neural_cleanse` is an educational implementation — results are illustrative and tuned for small demo models included in the repo.
- Privacy helpers are calculators (Laplace) and not a full DP training stack.

If you want, I can add CLI wrappers for `neural_cleanse` and `fairness` functions, or add example notebooks that step through the pipeline visually.
