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
FEATURE_NAMES = ["age", "insulin", "blood_glucose", "blood_pressure"]


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
    allowed_model_types: tuple[str, ...] = ("lin", "mlp"),
) -> pd.DataFrame:
    """
    Compute per-feature action diagnostics for a representative setup.
    We use epsilon=0.1 and seed=0 across available model/trainer configurations.
    """
    rows = []

    configs = (
        df[["model", "trainer"]]
        .drop_duplicates()
        .sort_values(["model", "trainer"])
        .itertuples(index=False)
    )
    for model_type, trainer_name in configs:
        if model_type not in allowed_model_types:
            continue
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

        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
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


def collect_action_instance_stats(
    df: pd.DataFrame,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    constraints: dict,
    epsilon: float = 0.1,
    seed: int = 0,
    allowed_model_types: tuple[str, ...] = ("lin", "mlp"),
) -> pd.DataFrame:
    """
    Collect per-instance action vectors and sparsity statistics for detailed diagnostics.
    """
    rows = []
    configs = (
        df[["model", "trainer"]]
        .drop_duplicates()
        .sort_values(["model", "trainer"])
        .itertuples(index=False)
    )
    for model_type, trainer_name in configs:
        if model_type not in allowed_model_types:
            continue
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
            actions, valids, costs, _, _ = recourse.causal_recourse(
                X_explain, explain, constraints, scm=scmm, verbose=False
            )
        else:
            hyperparams = utils.get_recourse_hyperparams(trainer_name)
            explain = recourse.DifferentiableRecourse(model, hyperparams)
            actions, valids, costs, _, _ = recourse.causal_recourse(
                X_explain,
                explain,
                constraints,
                scm=scmm,
                epsilon=epsilon,
                robust=epsilon > 0,
                verbose=False,
            )

        for i in range(actions.shape[0]):
            act = actions[i]
            nonzero = np.abs(act) > 1e-5
            rows.append(
                {
                    "model": model_type,
                    "trainer": trainer_name,
                    "config": f"{model_type.upper()}-{trainer_name}",
                    "epsilon": epsilon,
                    "seed": seed,
                    "instance_id": int(i),
                    "valid": bool(valids[i]),
                    "l0_nonzero": int(nonzero.sum()),
                    "l1_norm": float(np.abs(act).sum()),
                    "l2_norm": float(np.sqrt((act ** 2).sum())),
                    "cost": float(costs[i]),
                    "action_age": float(act[0]),
                    "action_insulin": float(act[1]),
                    "action_blood_glucose": float(act[2]),
                    "action_blood_pressure": float(act[3]),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "health_action_instance_stats.csv", index=False)
    return out


def summarize_sparsity(action_stats: pd.DataFrame) -> pd.DataFrame:
    if action_stats.empty:
        out = pd.DataFrame(
            columns=[
                "model",
                "trainer",
                "config",
                "epsilon",
                "seed",
                "valid_rate",
                "mean_l0_valid",
                "std_l0_valid",
                "mean_l1_valid",
                "std_l1_valid",
                "mean_l2_valid",
                "std_l2_valid",
            ]
        )
        out.to_csv(RESULTS_DIR / "health_sparsity_summary.csv", index=False)
        return out

    rows = []
    grp = action_stats.groupby(["model", "trainer", "config", "epsilon", "seed"], as_index=False)
    for (model, trainer, config, eps, seed), g in grp:
        valid_mask = g["valid"].astype(bool).to_numpy()
        valid_rate = float(valid_mask.mean()) if len(valid_mask) else 0.0
        gv = g[valid_mask] if valid_mask.any() else g
        rows.append(
            {
                "model": model,
                "trainer": trainer,
                "config": config,
                "epsilon": float(eps),
                "seed": int(seed),
                "valid_rate": valid_rate,
                "mean_l0_valid": float(gv["l0_nonzero"].mean()),
                "std_l0_valid": float(gv["l0_nonzero"].std(ddof=0)),
                "mean_l1_valid": float(gv["l1_norm"].mean()),
                "std_l1_valid": float(gv["l1_norm"].std(ddof=0)),
                "mean_l2_valid": float(gv["l2_norm"].mean()),
                "std_l2_valid": float(gv["l2_norm"].std(ddof=0)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "health_sparsity_summary.csv", index=False)
    return out


def collect_bootstrap_summary(
    instance_costs: pd.DataFrame, n_boot: int = 2000, seed: int = 123
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for valid_rate and valid_cost from instance-level recourse rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    grouped = instance_costs.groupby(["model", "trainer", "epsilon"], as_index=False)
    for (model, trainer, eps), g in grouped:
        valid = g["valid"].astype(bool).to_numpy()
        cost = g["cost"].to_numpy(dtype=float)
        n = len(g)
        if n == 0:
            continue

        vr_samples = np.zeros(n_boot, dtype=float)
        vc_samples = np.zeros(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            v = valid[idx]
            c = cost[idx]
            vr_samples[b] = v.mean() if v.size else 0.0
            vc_samples[b] = np.nanmean(c[v]) if np.any(v) else np.nan

        rows.append(
            {
                "model": model,
                "trainer": trainer,
                "config": f"{model.upper()}-{trainer}",
                "epsilon": float(eps),
                "valid_rate_mean": float(valid.mean()),
                "valid_rate_ci_low": float(np.nanpercentile(vr_samples, 2.5)),
                "valid_rate_ci_high": float(np.nanpercentile(vr_samples, 97.5)),
                "valid_cost_mean": float(np.nanmean(cost[valid])) if np.any(valid) else np.nan,
                "valid_cost_ci_low": float(np.nanpercentile(vc_samples, 2.5)),
                "valid_cost_ci_high": float(np.nanpercentile(vc_samples, 97.5)),
                "n_rows": int(n),
            }
        )

    out = pd.DataFrame(rows).sort_values(["model", "trainer", "epsilon"])
    out.to_csv(RESULTS_DIR / "health_bootstrap_summary.csv", index=False)
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
    action_instance_stats: pd.DataFrame,
    sparsity_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
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

    # Per-instance recourse costs (representative config: lin-ERM, eps=0.1, seed=0; fallback to first available).
    if not instance_costs.empty:
        rec = instance_costs[
            (instance_costs["model"] == "lin")
            & (instance_costs["trainer"] == "ERM")
            & (np.isclose(instance_costs["epsilon"], 0.1))
            & (instance_costs["seed"] == 0)
        ].sort_values("instance_id")
        if rec.empty:
            first_row = instance_costs.iloc[0]
            rec = instance_costs[
                (instance_costs["model"] == first_row["model"])
                & (instance_costs["trainer"] == first_row["trainer"])
                & (np.isclose(instance_costs["epsilon"], first_row["epsilon"]))
                & (instance_costs["seed"] == first_row["seed"])
            ].sort_values("instance_id")
        if not rec.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            colors = np.where(rec["valid"].astype(bool), "#2A9D8F", "#E76F51")
            ax.bar(rec["instance_id"], rec["cost"], color=colors, edgecolor="#1f2933")
            if rec["valid"].any():
                mean_cost = float(rec.loc[rec["valid"], "cost"].mean())
                ax.axhline(mean_cost, linestyle="--", linewidth=1.5, color="#264653", label=f"Mean valid cost: {mean_cost:.3f}")
                ax.legend(frameon=False)
            ax.set_title("Recourse Cost per Explained Instance (health, lin, ERM)")
            ax.set_xlabel("Explained Instance Index")
            ax.set_ylabel("L1 Recourse Cost")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(REPORT_FIG_DIR / "recourse_costs.png", dpi=220)
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

    # Bootstrap uncertainty bands for validity and cost curves.
    if not bootstrap_summary.empty:
        boot = bootstrap_summary.sort_values(["config", "epsilon"]).copy()
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
        for cfg, g in boot.groupby("config"):
            g = g.sort_values("epsilon")
            axes[0].plot(g["epsilon"], g["valid_rate_mean"], marker="o", label=cfg)
            axes[0].fill_between(
                g["epsilon"],
                g["valid_rate_ci_low"],
                g["valid_rate_ci_high"],
                alpha=0.18,
            )
            axes[1].plot(g["epsilon"], g["valid_cost_mean"], marker="o", label=cfg)
            axes[1].fill_between(
                g["epsilon"],
                g["valid_cost_ci_low"],
                g["valid_cost_ci_high"],
                alpha=0.18,
            )

        axes[0].set_title("Bootstrap CI: Validity")
        axes[0].set_xlabel("Epsilon")
        axes[0].set_ylabel("Valid Rate")
        axes[0].set_ylim(0.6, 1.02)
        axes[1].set_title("Bootstrap CI: Valid Cost")
        axes[1].set_xlabel("Epsilon")
        axes[1].set_ylabel("Valid L1 Cost")
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(REPORT_FIG_DIR / "bootstrap_ci_curves.png", dpi=220)
        plt.close(fig)

    # Sparsity-cost relation of valid recourse.
    if not sparsity_summary.empty:
        ss = sparsity_summary.copy()
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        sns.scatterplot(
            data=ss,
            x="mean_l0_valid",
            y="mean_l1_valid",
            hue="config",
            style="epsilon",
            s=95,
            ax=ax,
        )
        for _, r in ss.iterrows():
            ax.text(r["mean_l0_valid"] + 0.01, r["mean_l1_valid"] + 0.01, r["config"], fontsize=7)
        ax.set_title("Sparsity-Cost Operating Points (Valid Recourse)")
        ax.set_xlabel("Mean L0 (nonzero features)")
        ax.set_ylabel("Mean L1 cost")
        fig.tight_layout()
        fig.savefig(REPORT_FIG_DIR / "sparsity_vs_cost.png", dpi=220)
        plt.close(fig)

    # Instance-level action heatmap for an interpretable reference configuration.
    if not action_instance_stats.empty:
        hm = action_instance_stats[
            (action_instance_stats["model"] == "lin")
            & (action_instance_stats["trainer"] == "ERM")
            & (np.isclose(action_instance_stats["epsilon"], 0.1))
            & (action_instance_stats["seed"] == 0)
        ].sort_values("instance_id")
        if not hm.empty:
            mat = hm[
                ["action_age", "action_insulin", "action_blood_glucose", "action_blood_pressure"]
            ].to_numpy()
            row_labels = [f"id{int(i)}{'*' if v else ''}" for i, v in zip(hm["instance_id"], hm["valid"])]
            col_labels = FEATURE_NAMES

            fig, ax = plt.subplots(figsize=(8.2, 4.7))
            sns.heatmap(
                mat,
                cmap="coolwarm",
                center=0.0,
                xticklabels=col_labels,
                yticklabels=row_labels,
                ax=ax,
                cbar_kws={"label": "Action magnitude (signed)"},
            )
            ax.set_title("Action Heatmap (LIN-ERM, epsilon=0.1, seed=0; * = valid)")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Explained instance")
            fig.tight_layout()
            fig.savefig(REPORT_FIG_DIR / "action_heatmap_lin_erm.png", dpi=220)
            plt.close(fig)


def main():
    df, agg, X_train, Y_train, X_test, _, constraints = collect_metrics()
    if df.empty:
        raise RuntimeError("No result files found to build report artifacts.")

    full_diag = os.getenv("HW3_FULL_DIAGNOSTICS", "0") == "1"
    diag_models = ("lin", "mlp") if full_diag else ("lin",)

    nearest_vs_causal = nearest_vs_causal_lin(X_train, Y_train, X_test, constraints)
    instance_costs = collect_instance_costs(df)

    action_profiles_path = RESULTS_DIR / "health_action_profiles.csv"
    if action_profiles_path.exists():
        action_profiles = pd.read_csv(action_profiles_path)
    else:
        action_profiles = collect_action_profiles(
            df,
            X_train,
            Y_train,
            X_test,
            constraints,
            epsilon=0.1,
            seed=0,
            allowed_model_types=diag_models,
        )

    action_stats_path = RESULTS_DIR / "health_action_instance_stats.csv"
    if action_stats_path.exists():
        action_instance_stats = pd.read_csv(action_stats_path)
    else:
        action_instance_stats = collect_action_instance_stats(
            df,
            X_train,
            Y_train,
            X_test,
            constraints,
            epsilon=0.1,
            seed=0,
            allowed_model_types=diag_models,
        )
    sparsity_summary = summarize_sparsity(action_instance_stats)
    bootstrap_summary = collect_bootstrap_summary(instance_costs)
    make_figures(
        df,
        agg,
        nearest_vs_causal,
        instance_costs,
        action_profiles,
        action_instance_stats,
        sparsity_summary,
        bootstrap_summary,
    )
    print("Saved:", RESULTS_DIR / "health_report_summary.csv")
    print("Saved:", RESULTS_DIR / "health_report_aggregate.csv")
    print("Saved:", RESULTS_DIR / "nearest_vs_causal_lin_seed0.csv")
    print("Saved:", RESULTS_DIR / "health_instance_costs.csv")
    print("Saved:", RESULTS_DIR / "health_action_profiles.csv")
    print("Saved:", RESULTS_DIR / "health_action_instance_stats.csv")
    print("Saved:", RESULTS_DIR / "health_sparsity_summary.csv")
    print("Saved:", RESULTS_DIR / "health_bootstrap_summary.csv")
    print("Diagnostic model scope:", ",".join(diag_models), "(set HW3_FULL_DIAGNOSTICS=1 for lin+mlp)")
    print("Saved report figures in:", REPORT_FIG_DIR)
    print(agg.to_string(index=False))
    print(nearest_vs_causal.to_string(index=False))


if __name__ == "__main__":
    main()
