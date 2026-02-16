# HW4 — Security, Privacy & Fairness

Implementation of three trustworthiness subtopics with runnable demos and unit tests: backdoor detection (Neural Cleanse), differential privacy helpers, and fairness measurement/mitigation.

---

## High-level contents

- Security: trigger reconstruction, MAD outlier detector, unlearning demo (retraining on cleaned data).
- Privacy: Laplace mechanism helpers, composition utilities, example calculations for assignment tasks.
- Fairness: group metrics (disparate impact), proxy mitigation, promotion/demotion label swap technique + retraining.

---

## Quick setup & test

```bash
cd HomeWorks/HW4/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests
```

---

## Q1: Poisoned model setup (Security)

The security pipeline needs the attacked MNIST checkpoints. The **model index is chosen from the last digit of your student ID** (e.g. student ID `810101504` → model index **4** → `poisened_model_4.pth`).

You must do one of the following:

1. **Option A:** Place `poisened_models.rar` in `HomeWorks/HW4/code/` (or set `--archive-path` when running the generator). Install [The Unarchiver](https://theunarchiver.com/) / `unar` so the code can extract it. The script will extract into `code/model_weights/poisened_models/`.
2. **Option B:** Manually extract the archive so that `code/model_weights/poisened_models/` contains `poisened_model_0.pth` … `poisened_model_9.pth`.

Then run the report generator with your student ID so the correct checkpoint is used and report macros reflect your ID (see “Single-command run” below).

---

## Single-command run (generate report figures & metrics)

From `HomeWorks/HW4/code/`, after installing dependencies and placing or extracting the poisoned models:

```bash
python generate_report_figs.py --student-id YOUR_STUDENT_ID
```

Example: `--student-id 810101504` uses model index 4. You can also pass `--model-index 0` … `--model-index 9` directly.

If MNIST is not available online, use a local copy and disable download:

```bash
python generate_report_figs.py --student-id YOUR_STUDENT_ID --no-download-mnist
```

MNIST will be read from `code/data/mnist/` (or the path given by `--mnist-root`). Figures and LaTeX macros are written under `report/figures/` and `report/results/`.

---

## Where to look

- `code/neural_cleanse.py` — trigger optimization & detection pipeline.
- `code/privacy.py` — Laplace mechanism and composition helpers used in assignment questions.
- `code/fairness.py` — metric implementations and mitigation utilities.
- `code/model_weights/poisened_models/` — poisoned checkpoints (after extraction).

---

## Running the security demo

- Use the demo model included in `model_weights/` or supply your own model path to `neural_cleanse` helpers (see `code/neural_cleanse.py`).
- Example (from tests/demo): run `python generate_report_figs.py --student-id YOUR_STUDENT_ID` to produce Neural Cleanse figures and report macros.

---

## Notes & caveats

- The demo `neural_cleanse` implementation is pedagogical and for assignment evaluation; real backdoor analysis requires larger-scale evaluation and care about trigger transferability.
- Privacy helpers are didactic calculators for Laplace noise; they are not a full DP training framework.

If you want, I can also expand `code/README.md` with example commands and expected figure outputs or add additional unit tests for edge cases.
