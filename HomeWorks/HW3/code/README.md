# HW3/code/q5_codes — Developer reference

Detailed usage for the causal recourse implementation (training, SCM fitting, recourse solvers, and evaluation).

---

## Primary scripts (what to run)
- `train_classifiers.py` — train ERM / AF / regularized classifiers (saves to `models/`).
- `evaluate_recourse.py` — evaluate recourse validity & cost over chosen dataset and model.
- `runner.py` — experiment orchestration and batch evaluation across seeds/configs.
- `main.py` — simple entry point that wraps training + basic evaluation for homework deliverables.

---

## Example workflows
- Train and save a linear model (health dataset):
  ```bash
  python train_classifiers.py --dataset health --model lin --trainer ERM --seed 0 --save_model
  ```

- Compute recourse for N negative instances and report validity/cost:
  ```bash
  python evaluate_recourse.py --dataset health --model lin --trainer ERM --seed 0 --epsilon 0 --nexplain 10
  ```

- Run the default pipeline (quick demo):
  ```bash
  python main.py --seed 0
  ```

---

## Important implementation notes
- `LinearRecourse` will call `cvxpy` when available (recommended); otherwise a greedy fallback is used. Install `cvxpy` for exact convex solutions.
- Actionability constraints and feature metadata live in `data_utils.py` — update constraints here to change allowable interventions.
- SCMs: both hand-coded and learned SCM variants exist; `SCM_Trainer` can be used to fit structural equations from data.

---

## Outputs
- Models: `models/<dataset>_<model>_*.pth`
- SCMs: `scms/*.pth` and `.meta.json`
- Results: `results/*.csv` and `.npy` with recourse statistics

---

## Reproducibility & testing
- Use `--seed` to fix randomness.
- Scripts print summary `metrics.json`/CSV which are used by notebooks and the report.

---

## Tips & extensions
- Replace the default cost function in `recourse.py` to experiment with alternative user cost models.
- Add a new SCM by subclassing the generic `SCM` and adding it to `scm.py`.

Want me to add explicit `--help` outputs or example CLI commands for each script documented above? I can also add `Makefile` targets to standardize runs.