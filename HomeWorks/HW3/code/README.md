# HW3 Code Guide

The main implementation lives in `q5_codes/`.

## Key modules and methods

### 1) Data and constraints (`q5_codes/data_utils.py`)

- Loads `adult`, `compas`, `german`, and `health` datasets.
- For `health`, the loader defaults to `HomeWorks/HW3/dataset/diabetes.csv`
  (can be overridden via `HW3_HEALTH_DATA`), then maps columns to the
  required homework schema.
- Standardizes continuous features.
- Defines actionability constraints:
  - actionable feature indices
  - monotonic constraints (`increasing` / `decreasing`)
  - feasible feature limits

### 2) Classifiers and training (`q5_codes/trainers.py`)

- `LogisticRegression` and `MLP` classifiers.
- Threshold selection with maximum MCC (`set_max_mcc_threshold`).
- Trainer classes:
  - `ERM_Trainer`: standard risk minimization.
  - `Adversarial_Trainer`: FGSM/PGD-style adversarial updates.
  - `TRADES_Trainer`: KL-based robustness regularization.
  - `LLR_Trainer`: local linearity regularization (ALLR-style).
  - `Ross_Trainer`: actionable recourse-oriented regularizer.

### 3) Structural causal models (`q5_codes/scm.py`)

- Generic `SCM` class:
  - abduction-action-prediction style counterfactual generation
  - hard and soft interventions
  - Jacobian handling for intervention effects
- Implemented SCMs:
  - `SCM_Loan`
  - `Learned_Adult_SCM`
  - `Learned_COMPAS_SCM`
  - `Health_SCM`
- Includes structural-equation fitting helpers (`SCM_Trainer`, `MLP1`) for learned SCM variants.

### 4) Recourse algorithms (`q5_codes/recourse.py`)

- `LinearRecourse`:
  - solves constrained L1-minimization for linear classifiers.
  - uses `cvxpy` when available.
  - uses a deterministic greedy fallback solver when `cvxpy` is unavailable.
- `DifferentiableRecourse`:
  - gradient-based optimization of interventions for nonlinear models.
  - supports robust recourse via inner uncertainty maximization.
- `causal_recourse(...)`:
  - searches across actionable intervention sets and keeps the minimum-cost valid action.

### 5) End-to-end evaluation

- `train_classifiers.py`: trains and saves models.
- `evaluate_recourse.py`: computes recourse validity and cost.
- `runner.py`: benchmark orchestration.
- `main.py`: default HW entry point.

## Quick run

```bash
cd HomeWorks/HW3/code/q5_codes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --seed 0
```

## Useful direct commands

Train a linear classifier on health data:

```bash
python train_classifiers.py --dataset health --model lin --trainer ERM --seed 0 --save_model
```

Evaluate recourse:

```bash
python evaluate_recourse.py --dataset health --model lin --trainer ERM --seed 0 --epsilon 0 --nexplain 10
```

## Outputs

Depending on working directory, outputs are created under:

- `models/`
- `results/`
- `scms/`

## Notebook Deliverable

End-to-end HW3 notebook (Q1-Q6):

- `HomeWorks/HW3/code/HW3_complete_assignment.ipynb`
