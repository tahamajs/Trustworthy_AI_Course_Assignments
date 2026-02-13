# Trustworthy AI Course Assignments

This repository contains four course projects for Trustworthy AI.  
Each homework has runnable code, assignment text, and a report template.

## Repository map

- `HomeWorks/HW1`: generalization and robustness for image classification.
- `HomeWorks/HW2`: interpretability for tabular and vision models.
- `HomeWorks/HW3`: causal modeling and algorithmic recourse.
- `HomeWorks/HW4`: security (Neural Cleanse), privacy (differential privacy), and fairness.
- `template`: shared LaTeX templates.

## Methods implemented by project

- `HW1`: custom ResNet18, label smoothing, Circle Loss, FGSM/PGD attacks, adversarial training, UMAP.
- `HW2`: MLP and NAM classifiers, LIME, SHAP, Grad-CAM, Guided Backprop, SmoothGrad, activation maximization.
- `HW3`: SCM-based counterfactuals, linear and neural classifiers, ERM/AF/ALLR/ROSS trainers, linear and differentiable recourse.
- `HW4`: Neural Cleanse trigger reconstruction + MAD detection, unlearning by retraining, Laplace mechanism utilities, fairness metrics and promotion/demotion mitigation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install per-homework dependencies and run inside that homework:

- HW1: `pip install -r HomeWorks/HW1/code/requirements.txt`
- HW2: `pip install -r HomeWorks/HW2/code/requirements.txt`
- HW3: `pip install -r HomeWorks/HW3/code/q5_codes/requirements.txt`
- HW4: `pip install -r HomeWorks/HW4/code/requirements.txt`

## Notes

- Detailed run instructions are in each homework README and `code/README.md`.
- Some scripts support offline/demo fallbacks when datasets or pretrained weights are unavailable.
