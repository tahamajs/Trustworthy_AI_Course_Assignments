# Trustworthy_AI_Course_Assignments

A collection of course assignments and supporting code for the "Trustworthy AI" class — organized, documented, and ready for development or grading. This repository contains per-homework code, datasets, and assignment descriptions with consistent naming conventions and a recommended workflow.

---

## 📁 Repository structure

Top-level layout (important folders):

- `HomeWorks/` — all homework folders (HW1, HW2, ...), each with `code/`, `dataset/`, and `description/`.
- `template/` — report / LaTeX templates used for assignments.
- `.gitignore` — ignores virtualenvs, caches, editor files.

Example — `HomeWorks/HW3/`:
- `code/` — implementation and scripts (`q5_codes/` for question-specific code)
- `dataset/` — data files used by the exercises
- `description/README.md` — assignment text and PDFs
- `README.md` — short summary & how to run

---

## 🚀 Quick start — run a homework

1. Create a Python virtual environment and activate it:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r HomeWorks/HW3/code/Q5_codes/requirements.txt  # example
   ```

2. From the homework `code/` folder run the main script (example):

   ```bash
   cd HomeWorks/HW3/code/q5_codes
   python main.py
   ```

3. Check `results/` or `models/` folders inside the homework for outputs.

> Tip: each homework folder contains a `README.md` with homework-specific run instructions.

---

## 🧭 Conventions (what I applied)

- Filenames and folders: `snake_case`, lowercase.
- `description/en.md` renamed → `description/README.md` for consistency.
- Homework reports live under `HomeWorks/HW*/report/` and a `README.md` exists per HW.
- Branch for reorganizations: `reorganize/homeworks-structure` (already created).

---

## 🔧 Common git & workspace commands

- Create the recommended branch: `git checkout -b feature/your-change`
- Stage & commit: `git add . && git commit -m "<msg>"`
- Push: `git push -u origin <branch>`
- Open a PR from your branch when ready.

---

## ✅ Next recommended actions

- Populate each `code/` with entry-point scripts and clear `README.md` run examples.
- Move large datasets into `HomeWorks/HW*/dataset/` (already partly organized).
- Add unit tests under `tests/` if you plan automated checks.

---

## 📫 Questions / Changes

If you want me to:
- add run scripts for a specific HW, or
- move datasets into `dataset/`, or
- open a PR and squash/merge the reorganization branch —

tell me which homework to update and I’ll apply the changes.

---

Maintained by: repository owner — update `README.md` if you want a different layout or CI integration.
