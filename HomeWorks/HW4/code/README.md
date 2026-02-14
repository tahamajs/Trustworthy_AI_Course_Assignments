# HW4/code — Implementation & usage details

This document describes how to run the security, privacy, and fairness code, where test assets live, and how to reproduce the report figures.

---

## Key scripts & what they do
- `neural_cleanse.py` — trigger optimization + MAD-based target detection + unlearning demo.
- `privacy.py` — Laplace mechanism helpers and composition helpers used for assignment questions.
- `fairness.py` — fairness metrics and a simple promotion/demotion mitigation routine.
- `generate_report_figs.py` — builds PNG figures used in the LaTeX report.
- `tests/` — unit tests for each sub-module (run with `pytest`).

---

## Quick examples
- Run unit tests:
  ```bash
  pytest tests
  ```

- Generate all report figures (includes Neural Cleanse demo and fairness bars):
  ```bash
  python generate_report_figs.py
  ```

- Run Neural Cleanse pipeline on a supplied model:
  ```py
  # Example usage inside a Python session or script
  from neural_cleanse import reconstruct_trigger, detect_outlier_scales
  mask, pattern = reconstruct_trigger(model, data_loader)
  outlier_label = detect_outlier_scales(scales)
  ```

---

## Model weights & poisoned examples
- `model_weights/poisened_models/` contains example poisoned models used by unit tests — useful to validate the Neural Cleanse pipeline.

---

## Testing & validation
- The test suite verifies basic numerical properties and shapes and serves as a quick smoke-check for pipeline changes.
- Add tests in `tests/` when adding functionality; keep tests deterministic (fix seeds where randomness is used).

---

## Notes & limitations
- `neural_cleanse` is an educational implementation — results are illustrative and tuned for small demo models included in the repo.
- Privacy helpers are calculators (Laplace) and not a full DP training stack.

If you want, I can add CLI wrappers for `neural_cleanse` and `fairness` functions, or add example notebooks that step through the pipeline visually.