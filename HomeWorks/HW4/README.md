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

## Where to look
- `code/neural_cleanse.py` — trigger optimization & detection pipeline.
- `code/privacy.py` — Laplace mechanism and composition helpers used in assignment questions.
- `code/fairness.py` — metric implementations and mitigation utilities.
- `model_weights/poisened_models/` — example poisoned models used by tests.

---

## Running the security demo
- Use the demo model included in `model_weights/` or supply your own model path to `neural_cleanse` helpers (see `code/neural_cleanse.py`).
- Example (from tests/demo): run `python generate_report_figs.py` to produce a Neural Cleanse trigger figure used in the report.

---

## Notes & caveats
- The demo `neural_cleanse` implementation is pedagogical and for assignment evaluation; real backdoor analysis requires larger-scale evaluation and care about trigger transferability.
- Privacy helpers are didactic calculators for Laplace noise; they are not a full DP training framework.

If you want, I can also expand `code/README.md` with example commands and expected figure outputs or add additional unit tests for edge cases.
