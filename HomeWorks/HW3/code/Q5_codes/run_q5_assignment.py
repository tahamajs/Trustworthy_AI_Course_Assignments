"""
Run HW3 Q5 directly on the assignment dataset and save comparison artifacts.

This script executes the key Q5 steps with matched samples:
1) Load/process health data (mapped from dataset/diabetes.csv by default)
2) Train or load a linear ERM classifier
3) Evaluate recourse for the same 10 unhealthy individuals with:
   - SCM OFF (Nearest Counterfactual style)
   - SCM ON  (Causal Algorithmic Recourse)
4) Save summary and per-instance comparison CSVs
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

import data_utils
import recourse
import trainers
import train_classifiers
import utils


FEATURE_NAMES = ["age", "insulin", "blood_glucose", "blood_pressure"]


def _format_changed_features(action_row: np.ndarray, tol: float = 1e-6) -> str:
    changed = [FEATURE_NAMES[i] for i, v in enumerate(action_row) if abs(float(v)) > tol]
    return ",".join(changed) if changed else "none"


def run(seed: int = 0, n_explain: int = 10) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = "health"
    trainer_name = "ERM"
    model_type = "lin"
    lambd = utils.get_lambdas(dataset, model_type, trainer_name)

    source_path = data_utils.get_health_source_path()
    source_tag = data_utils.get_health_source_tag()

    # Ensure model exists for the currently selected source.
    model_path = utils.get_model_save_dir(dataset, trainer_name, model_type, seed, lambd) + ".pth"
    if not os.path.isfile(model_path):
        train_epochs = utils.get_train_epochs(dataset, model_type, trainer_name)
        train_classifiers.train(
            dataset,
            trainer_name,
            model_type,
            train_epochs,
            lambd,
            seed,
            verbose=True,
            save_model=True,
        )

    # Load split and model
    X, Y, constraints = data_utils.process_data(dataset)
    X_train, Y_train, X_test, _ = data_utils.train_test_split(X, Y)

    model = trainers.LogisticRegression(
        X_train.shape[-1],
        actionable_features=constraints["actionable"],
        actionable_mask=False,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.set_max_mcc_threshold(X_train, Y_train)

    # Match the same unhealthy individuals for both methods.
    id_neg = model.predict(X_test) == 0
    X_neg = X_test[id_neg]
    n = min(n_explain, X_neg.shape[0])
    if n == 0:
        raise RuntimeError("No unhealthy individuals were found in test split.")
    chosen = np.random.choice(np.arange(X_neg.shape[0]), size=n, replace=False)
    X_explain = X_neg[chosen]
    original_indices = np.argwhere(id_neg).reshape(-1)[chosen]

    # Recourse with SCM OFF (nearest counterfactual style).
    w, b = model.get_weights()
    explainer = recourse.LinearRecourse(w, b)
    a_off, valid_off, cost_off, _, _ = recourse.causal_recourse(
        X_explain, explainer, constraints, scm=None, verbose=False
    )
    valid_off = valid_off.astype(bool).reshape(-1)
    cost_off = np.asarray(cost_off).reshape(-1)

    # Recourse with SCM ON (causal recourse).
    scmm = utils.get_scm(model_type, dataset)
    a_on, valid_on, cost_on, _, _ = recourse.causal_recourse(
        X_explain, explainer, constraints, scm=scmm, verbose=False
    )
    valid_on = valid_on.astype(bool).reshape(-1)
    cost_on = np.asarray(cost_on).reshape(-1)

    # Summary table.
    summary = pd.DataFrame(
        [
            {
                "method": "nearest_counterfactual_scm_off",
                "valid_rate": float(valid_off.mean()),
                "mean_valid_cost": float(cost_off[valid_off].mean()) if valid_off.any() else np.nan,
                "n_explain": int(n),
            },
            {
                "method": "causal_recourse_scm_on",
                "valid_rate": float(valid_on.mean()),
                "mean_valid_cost": float(cost_on[valid_on].mean()) if valid_on.any() else np.nan,
                "n_explain": int(n),
            },
        ]
    )

    # Per-instance comparison table.
    per_instance_rows = []
    for i in range(n):
        per_instance_rows.append(
            {
                "instance_row_in_test": int(original_indices[i]),
                "valid_off": bool(valid_off[i]),
                "cost_off": float(cost_off[i]),
                "changed_features_off": _format_changed_features(a_off[i]),
                "valid_on": bool(valid_on[i]),
                "cost_on": float(cost_on[i]),
                "changed_features_on": _format_changed_features(a_on[i]),
                "cost_delta_off_minus_on": float(cost_off[i] - cost_on[i]),
            }
        )
    per_instance = pd.DataFrame(per_instance_rows)

    # Example instance report (Q5-F style).
    common_valid = np.where(valid_off & valid_on)[0]
    example_idx = int(common_valid[0]) if common_valid.size > 0 else 0
    example = pd.DataFrame(
        [
            {
                "method": "nearest_counterfactual_scm_off",
                "example_local_index": example_idx,
                "instance_row_in_test": int(original_indices[example_idx]),
                "valid": bool(valid_off[example_idx]),
                "cost": float(cost_off[example_idx]),
                "intervened_features": _format_changed_features(a_off[example_idx]),
                "action_age": float(a_off[example_idx][0]),
                "action_insulin": float(a_off[example_idx][1]),
                "action_blood_glucose": float(a_off[example_idx][2]),
                "action_blood_pressure": float(a_off[example_idx][3]),
            },
            {
                "method": "causal_recourse_scm_on",
                "example_local_index": example_idx,
                "instance_row_in_test": int(original_indices[example_idx]),
                "valid": bool(valid_on[example_idx]),
                "cost": float(cost_on[example_idx]),
                "intervened_features": _format_changed_features(a_on[example_idx]),
                "action_age": float(a_on[example_idx][0]),
                "action_insulin": float(a_on[example_idx][1]),
                "action_blood_glucose": float(a_on[example_idx][2]),
                "action_blood_pressure": float(a_on[example_idx][3]),
            },
        ]
    )

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "q5_diabetes_summary.csv")
    per_instance_path = os.path.join(out_dir, "q5_diabetes_per_instance.csv")
    example_path = os.path.join(out_dir, "q5_diabetes_example.csv")
    summary.to_csv(summary_path, index=False)
    per_instance.to_csv(per_instance_path, index=False)
    example.to_csv(example_path, index=False)

    print("Data source:", source_path)
    print("Data tag:", source_tag)
    print("Saved:", summary_path)
    print("Saved:", per_instance_path)
    print("Saved:", example_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nexplain", type=int, default=10)
    args = parser.parse_args()

    run(seed=args.seed, n_explain=args.nexplain)
