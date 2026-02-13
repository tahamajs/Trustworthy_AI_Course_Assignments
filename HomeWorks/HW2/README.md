# HW2: Interpretability (Tabular + Vision)

HW2 focuses on explaining model behavior for both tabular data and image models.

## Folders

- `code`: implementations for tabular and vision tasks.
- `notebooks`: homework notebook (`HW2_solution.ipynb`).
- `description`: official assignment description.
- `report`: report template and bibliography.

## Methods used in this project

- **Tabular modeling**:
  MLP classifier and Neural Additive Model (NAM).
- **Tabular explanation**:
  LIME (`LimeTabularExplainer`) and SHAP (`KernelExplainer`).
- **Vision explanation**:
  Grad-CAM, Guided Backpropagation, SmoothGrad, Guided Grad-CAM combinations.
- **Feature visualization**:
  activation maximization with total variation regularization and jitter.

## Quick start

```bash
cd HomeWorks/HW2/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tabular.py
```

For full step-by-step experiments, use:

- `HomeWorks/HW2/notebooks/HW2_solution.ipynb`
- `HomeWorks/HW2/code/README.md`
