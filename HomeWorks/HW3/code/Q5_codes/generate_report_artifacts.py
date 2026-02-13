"""Build HW3 report tables and figures from saved models/results."""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef

import data_utils
import recourse
import trainers
import utils


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORT_FIG_DIR = ROOT / "report" / "figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _parse_model_filename(path: Path):
    # Expected: health_<TRAINER>_<MODEL>_s<SEED>.pth
    m = re.match(r"health_([A-Za-z0-9]+)_(lin|mlp)_s(\d+)\.pth$", path.name)
    if m is None:
        return None
    trainer, model, seed = m.group(1), m.group(2), int(m.group(3))
    return trainer, model, seed


def _build_classifier(model_type: str, trainer_name: str, input_dim: int, actionable):
    if model_type == "lin":
        return trainers.LogisticRegression(
            input_dim,
            actionable_features=actionable,
            actionable_mask=(trainer_name == "AF"),
        )
    return trainers.MLP(
        input_dim,
        actionable_features=actionable,
        actionable_mask=(trainer_name == "AF"),
    )


def collect_metrics():
    X, Y, constraints = data_utils.process_data("health")
    X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)
    actionable = constraints["actionable"]

    rows = []
    for model_file in sorted(MODELS_DIR.glob("health_*_s*.pth")):
        parsed = _parse_model_filename(model_file)
        if parsed is None:
            continue
        trainer_name, model_type, seed = parsed
        lambd = utils.get_lambdas("health", model_type, trainer_name)

        model = _build_classifier(model_type, trainer_name, X_train.shape[-1], actionable)
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.set_max_mcc_threshold(X_train, Y_train)

        yhat = model.predict(X_test)
        acc = float((yhat == Y_test).mean())
        mcc = float(matthews_corrcoef(Y_test, yhat))

        for eps in [0.0, 0.1, 0.2]:
            save_base = Path(utils.get_metrics_save_dir("health", trainer_name, lambd, model_type, eps, seed))
            valid_path = Path(str(save_base) + "_valid.npy")
            cost_path = Path(str(save_base) + "_cost.npy")
            if not valid_path.exists() or not cost_path.exists():
                continue

            valid = np.load(valid_path).astype(bool)
            cost = np.load(cost_path)
            valid_rate = float(valid.mean()) if valid.size else 0.0
            valid_cost = float(cost[valid].mean()) if valid.any() else np.nan

            rows.append(
                {
                    "dataset": "health",
                    "model": model_type,
                    "trainer": trainer_name,
                    "seed": seed,
                    "epsilon": eps,
                    "accuracy": acc,
                    "mcc": mcc,
                    "valid_rate": valid_rate,
                    "valid_cost": valid_cost,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "health_report_summary.csv", index=False)

    agg = (
        df.groupby(["model", "trainer", "epsilon"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            valid_rate_mean=("valid_rate", "mean"),
            valid_rate_std=("valid_rate", "std"),
            valid_cost_mean=("valid_cost", "mean"),
            valid_cost_std=("valid_cost", "std"),
            runs=("seed", "nunique"),
        )
        .sort_values(["model", "trainer", "epsilon"])
    )
    agg.to_csv(RESULTS_DIR / "health_report_aggregate.csv", index=False)
    return df, agg, X_train, Y_train, X_test, Y_test, constraints


def collect_instance_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-instance valid/cost table from saved recourse arrays."""
    rows = []
    for _, row in df.iterrows():
        dataset = row["dataset"]
        trainer = row["trainer"]
        model_type = row["model"]
        seed = int(row["seed"])
        eps = float(row["epsilon"])
        lambd = utils.get_lambdas(dataset, model_type, trainer)
        save_base = Path(utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, eps, seed))
        valid_path = Path(str(save_base) + "_valid.npy")
        cost_path = Path(str(save_base) + "_cost.npy")
        if not valid_path.exists() or not cost_path.exists():
            continue

        valid = np.load(valid_path).astype(bool)
        cost = np.load(cost_path)
        for idx in range(cost.shape[0]):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_type,
                    "trainer": trainer,
                    "seed": seed,
                    "epsilon": eps,
                    "instance_id": idx,
                    "valid": bool(valid[idx]),
                    "cost": float(cost[idx]),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "health_instance_costs.csv", index=False)
    return out


def _build_model_for_eval(model_type: str, trainer_name: str, input_dim: int, actionable):
    if model_type == "lin":
        return trainers.LogisticRegression(
            input_dim,
            actionable_features=actionable,
            actionable_mask=(trainer_name == "AF"),
        )
    return trainers.MLP(
        input_dim,
        actionable_features=actionable,
        actionable_mask=(trainer_name == "AF"),
    )


def collect_action_profiles(
    df: pd.DataFrame,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    constraints: dict,
    epsilon: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute per-feature action diagnostics for a representative setup.
    We use epsilon=0.1 and seed=0 across available model/trainer configurations.
    """
    feature_names = ["age", "insulin", "blood_glucose", "blood_pressure"]
    rows = []

    configs = (
        df[["model", "trainer"]]
        .drop_duplicates()
        .sort_values(["model", "trainer"])
        .itertuples(index=False)
    )
    for model_type, trainer_name in configs:
        lambd = utils.get_lambdas("health", model_type, trainer_name)
        save_base = Path(utils.get_metrics_save_dir("health", trainer_name, lambd, model_type, epsilon, seed))
        id_path = Path(str(save_base) + "_ids.npy")
        model_path = MODELS_DIR / f"health_{trainer_name}_{model_type}_s{seed}.pth"
        if not id_path.exists() or not model_path.exists():
            continue

        model = _build_model_for_eval(model_type, trainer_name, X_train.shape[-1], constraints["actionable"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.set_max_mcc_threshold(X_train, Y_train)

        ids = np.load(id_path).reshape(-1).astype(int)
        if ids.size == 0:
            continue
        X_explain = X_test[ids]

        scmm = utils.get_scm(model_type, "health")
        if model_type == "lin":
            w, b = model.get_weights()
            Jw = w if scmm is None else scmm.get_Jacobian().T @ w
            dual_norm = np.sqrt(Jw.T @ Jw)
            explain = recourse.LinearRecourse(w, b + dual_norm * epsilon)
            actions, valids, _, _, _ = recourse.causal_recourse(
                X_explain, explain, constraints, scm=scmm, verbose=False
            )
        else:
            hyperparams = utils.get_recourse_hyperparams(trainer_name)
            explain = recourse.DifferentiableRecourse(model, hyperparams)
            actions, valids, _, _, _ = recourse.causal_recourse(
                X_explain,
                explain,
                constraints,
                scm=scmm,
                epsilon=epsilon,
                robust=epsilon > 0,
                verbose=False,
            )

        valid_mask = valids.astype(bool)
        if not valid_mask.any():
            valid_mask = np.ones(actions.shape[0], dtype=bool)

        for feat_idx, feat_name in enumerate(feature_names):
            abs_all = np.abs(actions[:, feat_idx])
            abs_valid = np.abs(actions[valid_mask, feat_idx])
            rows.append(
                {
                    "model": model_type,
                    "trainer": trainer_name,
                    "config": f"{model_type.upper()}-{trainer_name}",
                    "epsilon": epsilon,
                    "seed": seed,
                    "feature": feat_name,
                    "mean_abs_action_all": float(abs_all.mean()),
                    "mean_abs_action_valid": float(abs_valid.mean()),
                    "nonzero_rate_all": float((abs_all > 1e-5).mean()),
                    "nonzero_rate_valid": float((abs_valid > 1e-5).mean()),
                    "actionable": int(feat_idx in constraints["actionable"]),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "health_action_profiles.csv", index=False)
    return out


def nearest_vs_causal_lin(X_train, Y_train, X_test, constraints):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_file = MODELS_DIR / "health_ERM_lin_s0.pth"
    model = trainers.LogisticRegression(
        X_train.shape[-1],
        actionable_features=constraints["actionable"],
        actionable_mask=False,
    )
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.set_max_mcc_threshold(X_train, Y_train)

    id_neg = model.predict(X_test) == 0
    X_neg = X_test[id_neg]
    n_explain = min(10, len(X_neg))
    idx = np.random.choice(np.arange(X_neg.shape[0]), size=n_explain, replace=False)
    X_explain = X_neg[idx]

    w, b = model.get_weights()
    explain = recourse.LinearRecourse(w, b)

    _, valid_n, cost_n, _, _ = recourse.causal_recourse(
        X_explain,
        explain,
        constraints,
        scm=None,
        verbose=False,
    )
    _, valid_c, cost_c, _, _ = recourse.causal_recourse(
        X_explain,
        explain,
        constraints,
        scm=utils.get_scm("lin", "health"),
        verbose=False,
    )

    records = [
        {
            "method": "Nearest Counterfactual (SCM off)",
            "valid_rate": float(valid_n.mean()) if valid_n.size else 0.0,
            "valid_cost": float(cost_n[valid_n].mean()) if valid_n.any() else np.nan,
        },
        {
            "method": "Causal Algorithmic Recourse (SCM on)",
            "valid_rate": float(valid_c.mean()) if valid_c.size else 0.0,
            "valid_cost": float(cost_c[valid_c].mean()) if valid_c.any() else np.nan,
        },
    ]
    out = pd.DataFrame(records)
    out.to_csv(RESULTS_DIR / "nearest_vs_causal_lin_seed0.csv", index=False)
    return out


def make_figures(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    nearest_vs_causal: pd.DataFrame,
    instance_costs: pd.DataFrame,
    action_profiles: pd.DataFrame,
):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    df = df.copy()
    df["config"] = df["model"].str.upper() + "-" + df["trainer"]

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    sns.lineplot(
        data=agg.assign(config=lambda d: d["model"].str.upper() + "-" + d["trainer"]),
        x="epsilon",
        y="valid_rate_mean",
        hue="config",
        marker="o",
        ax=ax,
    )
    ax.set_title("Recourse Validity vs Robustness Radius")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Mean Valid Recourse Rate")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "valid_rate_vs_epsilon.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    sns.lineplot(
        data=agg.assign(config=lambda d: d["model"].str.upper() + "-" + d["trainer"]),
        x="epsilon",
        y="valid_cost_mean",
        hue="config",
        marker="o",
        ax=ax,
    )
    ax.set_title("Cost of Valid Recourse vs Robustness Radius")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Mean Valid L1 Cost")
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "valid_cost_vs_epsilon.png", dpi=220)
    plt.close(fig)

    clf = (
        df[df["epsilon"] == 0.0][["model", "trainer", "seed", "accuracy", "mcc"]]
        .drop_duplicates()
        .assign(config=lambda d: d["model"].str.upper() + "-" + d["trainer"])
    )
    clf_melt = clf.melt(id_vars=["config", "seed"], value_vars=["accuracy", "mcc"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(data=clf_melt, x="config", y="value", hue="metric", errorbar="sd", ax=ax)
    ax.set_title("Classifier Quality (Health Dataset)")
    ax.set_xlabel("Model / Trainer")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "classifier_metrics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    melt = nearest_vs_causal.melt(id_vars=["method"], value_vars=["valid_rate", "valid_cost"], var_name="metric", value_name="value")
    sns.barplot(data=melt, x="method", y="value", hue="metric", ax=ax)
    ax.set_title("Nearest Counterfactual vs Causal Recourse (lin-ERM, seed=0)")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "nearest_vs_causal.png", dpi=220)
    plt.close(fig)

    # Validity-cost operating points.
    scatter_df = agg.assign(config=lambda d: d["model"].str.upper() + "-" + d["trainer"]).copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    sns.scatterplot(
        data=scatter_df,
        x="valid_rate_mean",
        y="valid_cost_mean",
        hue="config",
        style="epsilon",
        s=95,
        ax=ax,
    )
    ax.set_title("Validity-Cost Frontier (Health)")
    ax.set_xlabel("Mean Valid Recourse Rate")
    ax.set_ylabel("Mean Valid L1 Cost")
    ax.set_xlim(0.65, 1.02)
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "validity_cost_frontier.png", dpi=220)
    plt.close(fig)

    # Distribution of per-instance costs.
    if not instance_costs.empty:
        dist_df = instance_costs.copy()
        dist_df["config"] = dist_df["model"].str.upper() + "-" + dist_df["trainer"]
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        sns.boxplot(
            data=dist_df,
            x="config",
            y="cost",
            hue="epsilon",
            showfliers=False,
            ax=ax,
        )
        ax.set_title("Per-Instance Recourse Cost Distribution")
        ax.set_xlabel("Model / Trainer")
        ax.set_ylabel("L1 Cost")
        ax.tick_params(axis="x", labelrotation=12)
        fig.tight_layout()
        fig.savefig(REPORT_FIG_DIR / "cost_distribution_boxplot.png", dpi=220)
        plt.close(fig)

    # Per-feature action intensity and activation frequency.
    if not action_profiles.empty:
        prof = action_profiles.copy()
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        sns.barplot(
            data=prof,
            x="feature",
            y="mean_abs_action_valid",
            hue="config",
            ax=ax,
        )
        ax.set_title("Feature-wise Mean Absolute Action (epsilon=0.1, seed=0)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean |Action| (valid recourse)")
        fig.tight_layout()
        fig.savefig(REPORT_FIG_DIR / "feature_mean_abs_action.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        sns.barplot(
            data=prof,
            x="feature",
            y="nonzero_rate_valid",
            hue="config",
            ax=ax,
        )
        ax.set_title("Feature Intervention Activation Rate (epsilon=0.1, seed=0)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("P(|Action| > 1e-5 | valid)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(REPORT_FIG_DIR / "feature_nonzero_rate.png", dpi=220)
        plt.close(fig)


def main():
    df, agg, X_train, Y_train, X_test, _, constraints = collect_metrics()
    if df.empty:
        raise RuntimeError("No result files found to build report artifacts.")
    nearest_vs_causal = nearest_vs_causal_lin(X_train, Y_train, X_test, constraints)
    instance_costs = collect_instance_costs(df)
    action_profiles = collect_action_profiles(df, X_train, Y_train, X_test, constraints, epsilon=0.1, seed=0)
    make_figures(df, agg, nearest_vs_causal, instance_costs, action_profiles)
    print("Saved:", RESULTS_DIR / "health_report_summary.csv")
    print("Saved:", RESULTS_DIR / "health_report_aggregate.csv")
    print("Saved:", RESULTS_DIR / "nearest_vs_causal_lin_seed0.csv")
    print("Saved:", RESULTS_DIR / "health_instance_costs.csv")
    print("Saved:", RESULTS_DIR / "health_action_profiles.csv")
    print("Saved report figures in:", REPORT_FIG_DIR)
    print(agg.to_string(index=False))
    print(nearest_vs_causal.to_string(index=False))


if __name__ == "__main__":
    main()
