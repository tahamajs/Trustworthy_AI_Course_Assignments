# HW4 Code Guide

This folder contains reference implementations for all HW4 questions.

## Implemented methods

### 1) Security (`neural_cleanse.py`)

- `reconstruct_trigger(...)`
  - Optimizes a trigger mask and pattern for each target label.
  - Loss = classification to target label + L1(mask) + pattern regularization.
- `detect_outlier_scales(...)`
  - MAD-based anomaly score across reconstructed trigger scales.
  - Returns the most suspicious target label.
- `unlearn_by_retraining(...)`
  - Applies trigger to a data fraction and retrains for one/few epochs.
- `DemoConvNet` and `load_model(...)`
  - Demo model path so the pipeline can run without external checkpoints.

### 2) Privacy (`privacy.py`)

- `laplace_scale(...)`: computes `b = sensitivity / epsilon`.
- `add_laplace_noise(...)`: sampled noisy response.
- `laplace_cdf_threshold(...)`: probability query for noisy thresholds.
- `compose_epsilons(...)`: basic sequential composition.
- `average_income_scale(...)`: convenience helper for assignment-style examples.

### 3) Fairness (`fairness.py`)

- `accuracy(...)`: standard prediction accuracy.
- `disparate_impact(...)`: ratio of positive prediction rates across groups.
- `zemel_proxy_fairness(...)`: cluster-based fairness proxy (Zemel-inspired).
- `train_baseline_model(...)`: scaled logistic regression baseline.
- `apply_promotion_demotion(...)`: assignment rule for selecting label swaps.
- `retrain_with_swapped_labels(...)`: retrains on debiased targets.

### 4) Reporting helper (`generate_report_figs.py`)

- Generates fairness comparison bars, Neural Cleanse trigger figure, and prints privacy example values.

### 5) Tests (`tests/`)

- `test_fairness.py`: metric and promotion/demotion checks.
- `test_privacy.py`: Laplace helper checks.
- `test_neural_cleanse.py`: shape/outlier checks for security pipeline.

## Quick run

```bash
cd HomeWorks/HW4/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests
```

Run figure generation:

```bash
python generate_report_figs.py
```

## Notes

- For real backdoor analysis, load attacked weights through `load_model(path=...)`.
- Demo mode in `neural_cleanse.py` is intended for workflow validation, not final accuracy claims.
