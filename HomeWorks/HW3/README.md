# HW3: Causal Recourse

HW3 focuses on causal modeling and algorithmic recourse.

## Folders

- `code/q5_codes`: full implementation for training and recourse evaluation.
- `dataset/diabetes.csv`: primary dataset used for health-related questions (Q4/Q5),
  mapped to `age/insulin/blood_glucose/blood_pressure/category` in code.
- `dataset`: additional dataset archives.
- `description`: assignment statement.
- `report`: report template.

## Methods used in this project

- **Data processing and constraints**:
  dataset-specific preprocessing + actionability constraints (`data_utils.py`).
- **Predictive models**:
  logistic regression and MLP classifiers with threshold calibration by MCC (`trainers.py`).
- **Training strategies**:
  ERM, actionable-feature masking (AF), ALLR/LLR-style regularization, Ross regularizer.
- **Causal modeling**:
  structural causal model base class + dataset SCMs (Loan, Adult, COMPAS, Health).
- **Recourse methods**:
  linear recourse via constrained optimization and differentiable recourse via iterative gradient-based optimization.
- **Causal recourse evaluation**:
  validity and cost metrics over negatively classified instances.

## Quick start

```bash
cd HomeWorks/HW3/code/q5_codes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --seed 0
```

Detailed method-level documentation is in:

- `HomeWorks/HW3/code/README.md`

## Complete Notebook

A full notebook that covers all HW3 parts (Q1-Q6), including runnable Q5 causal recourse pipeline on the provided dataset folder, is available at:

- `HomeWorks/HW3/code/HW3_complete_assignment.ipynb`
- `HomeWorks/HW3/output/jupyter-notebook/hw3_complete_assignment.ipynb`
