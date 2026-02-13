# HW4: Security, Privacy, and Fairness

HW4 implements three trustworthiness topics:

- backdoor detection and mitigation (security),
- differential privacy calculations (privacy),
- bias measurement and mitigation (fairness).

## Folders

- `code`: all runnable scripts and tests.
- `description`: assignment statement.
- `report`: report template.
- `ZipFile`: archived homework submission material.

## Methods used in this project

- **Security**:
  Neural Cleanse-style trigger reconstruction, MAD-based outlier label detection, unlearning via retraining.
- **Privacy**:
  Laplace mechanism scale/noise/probability calculations and epsilon composition helpers.
- **Fairness**:
  accuracy, disparate impact, Zemel-style proxy fairness, promotion/demotion label-swapping mitigation with retraining.

## Quick start

```bash
cd HomeWorks/HW4/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests
```

Detailed method-level documentation is in:

- `HomeWorks/HW4/code/README.md`
