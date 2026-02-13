from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from fairness import (
    apply_group_thresholds,
    apply_promotion_demotion,
    compute_fairness_metrics,
    load_data,
    optimize_group_thresholds,
    retrain_with_swapped_labels,
    train_baseline_model,
    train_reweighed_model,
)
from neural_cleanse import (
    apply_trigger,
    detect_outlier_scales,
    evaluate_asr,
    evaluate_clean_accuracy,
    load_mnist_test,
    load_model,
    reconstruct_all_labels,
    resolve_checkpoint_path,
    unlearn_by_retraining,
)
from privacy import counting_query_results, income_query_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW4 report artifacts.")
    parser.add_argument("--student-id", type=str, default="810101504")
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--archive-path", type=str, default=None)
    parser.add_argument("--extract-dir", type=str, default=None)
    parser.add_argument("--mnist-root", type=str, default=None)
    parser.add_argument("--download-mnist", action="store_true", dest="download_mnist")
    parser.add_argument("--no-download-mnist", action="store_false", dest="download_mnist")
    parser.set_defaults(download_mnist=True)
    parser.add_argument(
        "--security-profile",
        type=str,
        default="high-fidelity",
        choices=["fast-smoke", "balanced", "high-fidelity"],
    )
    parser.add_argument("--swap-k", type=int, default=10)
    parser.add_argument("--population-n", type=int, default=500)
    parser.add_argument("--unbounded-p", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_paths() -> Dict[str, Path]:
    code_dir = Path(__file__).resolve().parent
    root_dir = code_dir.parent
    report_dir = root_dir / "report"
    figures_dir = report_dir / "figures"
    results_dir = report_dir / "results"
    model_root = code_dir / "model_weights"
    mnist_root = code_dir / "data" / "mnist"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    mnist_root.mkdir(parents=True, exist_ok=True)
    return {
        "code_dir": code_dir,
        "root_dir": root_dir,
        "report_dir": report_dir,
        "figures_dir": figures_dir,
        "results_dir": results_dir,
        "model_root": model_root,
        "mnist_root": mnist_root,
    }


def configure_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def profile_config(profile: str) -> Dict[str, int | None]:
    if profile == "fast-smoke":
        return {"steps": 120, "batch_size": 64, "limit": 1000}
    if profile == "balanced":
        return {"steps": 250, "batch_size": 128, "limit": 3000}
    return {"steps": 500, "batch_size": 128, "limit": None}


def infer_expected_attacked_label(checkpoint_path: str) -> int | None:
    m = re.search(r"poisened_model_(\d+)\.pth$", checkpoint_path)
    if m:
        return int(m.group(1))
    return None


def run_security(
    args: argparse.Namespace,
    paths: Dict[str, Path],
) -> Tuple[Dict[str, float | int], Dict[str, str]]:
    conf = profile_config(args.security_profile)
    archive_path = args.archive_path or str(paths["code_dir"] / "poisened_models.rar")
    extract_dir = args.extract_dir or str(paths["model_root"])
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = resolve_checkpoint_path(
            student_id=args.student_id,
            model_index=args.model_index,
            archive_path=archive_path,
            extract_dir=extract_dir,
        )

    mnist_root = args.mnist_root or str(paths["mnist_root"])
    loader = load_mnist_test(
        root=mnist_root,
        download=args.download_mnist,
        batch_size=int(conf["batch_size"]),
        limit=conf["limit"],
        seed=args.seed,
    )
    eval_loader = DataLoader(loader.dataset, batch_size=int(conf["batch_size"]), shuffle=False, num_workers=0)

    model = load_model(path=checkpoint_path, device="cpu")
    model_unlearn = copy.deepcopy(model)

    recon = reconstruct_all_labels(
        model=model,
        dataloader=loader,
        num_classes=10,
        device="cpu",
        steps=int(conf["steps"]),
        lr=0.1,
    )
    scales = [recon[label].scale for label in range(10)]
    attacked_label = detect_outlier_scales(scales)
    attacked_trigger = recon[attacked_label]

    trigger_mask_path = paths["figures_dir"] / "trigger_reconstructed.png"
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(attacked_trigger.mask.squeeze().numpy(), cmap="gray")
    plt.title(f"Reconstructed mask (label={attacked_label})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(trigger_mask_path, dpi=180)
    plt.close()

    trigger_grid_path = paths["figures_dir"] / "trigger_all_labels_grid.png"
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
    for label in range(10):
        ax = axes.flat[label]
        ax.imshow(recon[label].mask.squeeze().numpy(), cmap="gray")
        ax.set_title(f"y={label}, s={recon[label].scale:.1f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Neural Cleanse reconstruction across labels", fontsize=11)
    plt.tight_layout()
    plt.savefig(trigger_grid_path, dpi=180)
    plt.close(fig)

    def trigger_fn(x: torch.Tensor) -> torch.Tensor:
        return apply_trigger(x, attacked_trigger.mask, attacked_trigger.pattern)

    clean_acc_before = evaluate_clean_accuracy(model, eval_loader, device="cpu")
    asr_before = evaluate_asr(model, eval_loader, trigger_fn, attacked_label, device="cpu")

    unlearn_by_retraining(
        model=model_unlearn,
        dataset=loader.dataset,
        trigger_fn=trigger_fn,
        fraction=0.2,
        epochs=1,
        lr=1e-3,
        batch_size=int(conf["batch_size"]),
        device="cpu",
        seed=args.seed,
    )

    clean_acc_after = evaluate_clean_accuracy(model_unlearn, eval_loader, device="cpu")
    asr_after = evaluate_asr(model_unlearn, eval_loader, trigger_fn, attacked_label, device="cpu")

    security_bar_path = paths["figures_dir"] / "security_before_after.png"
    metrics = ["Clean Accuracy", "ASR"]
    before_vals = [clean_acc_before, asr_before]
    after_vals = [clean_acc_after, asr_after]
    x = np.arange(len(metrics))
    w = 0.35
    fig2, ax2 = plt.subplots(figsize=(6.0, 3.5))
    ax2.bar(x - w / 2, before_vals, width=w, label="Before unlearning")
    ax2.bar(x + w / 2, after_vals, width=w, label="After unlearning")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    for i in range(len(metrics)):
        ax2.text(x[i] - w / 2, before_vals[i] + 0.02, f"{before_vals[i]:.3f}", ha="center", fontsize=8)
        ax2.text(x[i] + w / 2, after_vals[i] + 0.02, f"{after_vals[i]:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(security_bar_path, dpi=180)
    plt.close(fig2)

    expected_label = infer_expected_attacked_label(checkpoint_path)

    summary = {
        "checkpoint_path": checkpoint_path,
        "expected_attacked_label": expected_label if expected_label is not None else -1,
        "detected_attacked_label": int(attacked_label),
        "clean_accuracy_before": float(clean_acc_before),
        "asr_before": float(asr_before),
        "clean_accuracy_after": float(clean_acc_after),
        "asr_after": float(asr_after),
    }
    figure_paths = {
        "trigger_reconstructed": str(trigger_mask_path),
        "trigger_all_labels_grid": str(trigger_grid_path),
        "security_before_after": str(security_bar_path),
    }
    return summary, figure_paths


def run_privacy(args: argparse.Namespace, paths: Dict[str, Path]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    income = income_query_results(
        epsilon=0.1,
        sensitivity_avg=5000.0,
        sensitivity_total=50000.0,
        true_avg=40000.0,
        true_total=20_000_000.0,
        sampled_noise_avg=2000.0,
        sampled_noise_total=5000.0,
        epsilon_avg_split=0.05,
        epsilon_total_split=0.05,
    )
    counting = counting_query_results(
        epsilon=0.1,
        delta=1e-5,
        sensitivity=1.0,
        true_value=500.0,
        threshold=505.0,
        k=92,
        n=args.population_n,
        p=args.unbounded_p,
    )

    privacy_plot = paths["figures_dir"] / "privacy_scenarios.png"
    scenario_names = ["Base", "Sequential", "Unbounded"]
    b_values = [counting["b_base"], counting["b_sequential"], counting["b_unbounded"]]
    p_values = [
        counting["prob_base_gt_threshold"],
        counting["prob_sequential_gt_threshold"],
        counting["prob_unbounded_gt_threshold"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].bar(scenario_names, b_values, color=["#4e79a7", "#f28e2c", "#59a14f"])
    axes[0].set_title("Laplace scale b")
    axes[0].set_ylabel("b")

    axes[1].bar(scenario_names, p_values, color=["#4e79a7", "#f28e2c", "#59a14f"])
    axes[1].set_title("P(noisy > 505)")
    axes[1].set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(privacy_plot, dpi=180)
    plt.close(fig)

    return {"income": income, "counting": counting}, {"privacy_scenarios": str(privacy_plot)}


def run_fairness(args: argparse.Namespace, paths: Dict[str, Path]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    data_path = paths["code_dir"] / "data.csv"
    Xdf, y = load_data(str(data_path))
    y_np = y.to_numpy()
    sensitive = Xdf["gender"].to_numpy()

    Xnum = Xdf.select_dtypes(include=[np.number])
    columns = list(Xnum.columns)
    sens_col_idx = columns.index("gender")
    X_with_gender = Xnum.to_numpy()

    idx_all = np.arange(X_with_gender.shape[0])
    idx_train, idx_test = train_test_split(idx_all, test_size=0.3, random_state=0)
    X_train = X_with_gender[idx_train]
    X_test = X_with_gender[idx_test]
    y_train = y_np[idx_train]
    y_test = y_np[idx_test]
    sens_train = sensitive[idx_train]
    sens_test = sensitive[idx_test]

    # Base model
    clf_base = train_baseline_model(X_train, y_train)
    Xs_train_base = clf_base._scaler.transform(X_train)
    Xs_test_base = clf_base._scaler.transform(X_test)
    proba_train_base = clf_base.predict_proba(Xs_train_base)[:, 1]
    pred_train_base = (proba_train_base >= 0.5).astype(int)
    proba_test_base = clf_base.predict_proba(Xs_test_base)[:, 1]
    pred_test_base = (proba_test_base >= 0.5).astype(int)

    fairness_metrics: Dict[str, Dict[str, float]] = {}
    fairness_metrics["baseline"] = compute_fairness_metrics(Xs_test_base, y_test, pred_test_base, sens_test)

    # Assignment mitigation (promotion/demotion)
    swap_mask = apply_promotion_demotion(
        y_proba=proba_train_base,
        y_pred=pred_train_base,
        sensitive=sens_train,
        k=args.swap_k,
    )
    clf_swap = retrain_with_swapped_labels(X_train, y_train, swap_mask)
    Xs_test_swap = clf_swap._scaler.transform(X_test)
    pred_test_swap = (clf_swap.predict_proba(Xs_test_swap)[:, 1] >= 0.5).astype(int)
    fairness_metrics["promotion_demotion"] = compute_fairness_metrics(Xs_test_swap, y_test, pred_test_swap, sens_test)
    fairness_metrics["promotion_demotion"]["swap_count"] = float(swap_mask.sum())

    # Sensitive-feature removal baseline
    X_no_gender = np.delete(X_with_gender, sens_col_idx, axis=1)
    Xng_train = X_no_gender[idx_train]
    Xng_test = X_no_gender[idx_test]
    clf_no_gender = train_baseline_model(Xng_train, y_train)
    Xngs_test = clf_no_gender._scaler.transform(Xng_test)
    pred_no_gender = (clf_no_gender.predict_proba(Xngs_test)[:, 1] >= 0.5).astype(int)
    fairness_metrics["no_gender"] = compute_fairness_metrics(Xngs_test, y_test, pred_no_gender, sens_test)

    # Bonus method 1: reweighing
    clf_reweighed = train_reweighed_model(X_train, y_train, sens_train)
    Xs_test_rw = clf_reweighed._scaler.transform(X_test)
    pred_rw = (clf_reweighed.predict_proba(Xs_test_rw)[:, 1] >= 0.5).astype(int)
    fairness_metrics["reweighed"] = compute_fairness_metrics(Xs_test_rw, y_test, pred_rw, sens_test)

    # Bonus method 2: group thresholds
    thresholds = optimize_group_thresholds(y_true=y_train, y_proba=proba_train_base, sensitive=sens_train)
    pred_group_thr = apply_group_thresholds(proba_test_base, sens_test, thresholds)
    fairness_metrics["group_thresholds"] = compute_fairness_metrics(Xs_test_base, y_test, pred_group_thr, sens_test)
    fairness_metrics["group_thresholds"]["threshold_group_0"] = float(thresholds[0])
    fairness_metrics["group_thresholds"]["threshold_group_1"] = float(thresholds[1])

    plot_path = paths["figures_dir"] / "fairness_comparison.png"
    model_order = ["baseline", "promotion_demotion", "no_gender", "reweighed", "group_thresholds"]
    display = ["Base", "Swap", "No-gender", "Reweigh", "Grp-thr"]

    acc_vals = [fairness_metrics[k]["accuracy"] for k in model_order]
    di_vals = [fairness_metrics[k]["disparate_impact"] for k in model_order]
    zem_vals = [fairness_metrics[k]["zemel_proxy"] for k in model_order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(display, acc_vals, color="#4e79a7")
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(display, di_vals, color="#f28e2c")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Disparate Impact")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(display, zem_vals, color="#59a14f")
    axes[2].set_title("Zemel proxy (lower is better)")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)
    return fairness_metrics, {"fairness_comparison": str(plot_path)}


def write_results(paths: Dict[str, Path], payload: Dict) -> Tuple[str, str]:
    metrics_path = paths["results_dir"] / "metrics_summary.json"
    macros_path = paths["results_dir"] / "results_macros.tex"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    sec = payload.get("security", {})
    prv = payload.get("privacy", {})
    fair = payload.get("fairness", {})

    def metric(model_key: str, metric_key: str, default: float = -1.0) -> float:
        return float(fair.get(model_key, {}).get(metric_key, default))

    lines = [
        "% Auto-generated by code/generate_report_figs.py",
        "\\newcommand{\\QOneDetectedLabel}{" + str(sec.get("detected_attacked_label", -1)) + "}",
        "\\newcommand{\\QOneExpectedLabel}{" + str(sec.get("expected_attacked_label", -1)) + "}",
        "\\newcommand{\\QOneCleanAccBefore}{" + f"{float(sec.get('clean_accuracy_before', -1.0)):.4f}" + "}",
        "\\newcommand{\\QOneAsrBefore}{" + f"{float(sec.get('asr_before', -1.0)):.4f}" + "}",
        "\\newcommand{\\QOneCleanAccAfter}{" + f"{float(sec.get('clean_accuracy_after', -1.0)):.4f}" + "}",
        "\\newcommand{\\QOneAsrAfter}{" + f"{float(sec.get('asr_after', -1.0)):.4f}" + "}",
        "\\newcommand{\\QTwoBBase}{" + f"{float(prv.get('counting', {}).get('b_base', -1.0)):.4f}" + "}",
        "\\newcommand{\\QTwoBSequential}{" + f"{float(prv.get('counting', {}).get('b_sequential', -1.0)):.4f}" + "}",
        "\\newcommand{\\QTwoBUnbounded}{" + f"{float(prv.get('counting', {}).get('b_unbounded', -1.0)):.4f}" + "}",
        "\\newcommand{\\QTwoProbBase}{" + f"{float(prv.get('counting', {}).get('prob_base_gt_threshold', -1.0)):.4f}" + "}",
        "\\newcommand{\\QTwoProbSequential}{" + f"{float(prv.get('counting', {}).get('prob_sequential_gt_threshold', -1.0)):.4f}" + "}",
        "\\newcommand{\\QTwoProbUnbounded}{" + f"{float(prv.get('counting', {}).get('prob_unbounded_gt_threshold', -1.0)):.4f}" + "}",
        "\\newcommand{\\QThreeBaseAcc}{" + f"{metric('baseline', 'accuracy'):.4f}" + "}",
        "\\newcommand{\\QThreeBaseDi}{" + f"{metric('baseline', 'disparate_impact'):.4f}" + "}",
        "\\newcommand{\\QThreeBaseZemel}{" + f"{metric('baseline', 'zemel_proxy'):.4f}" + "}",
        "\\newcommand{\\QThreeSwapAcc}{" + f"{metric('promotion_demotion', 'accuracy'):.4f}" + "}",
        "\\newcommand{\\QThreeSwapDi}{" + f"{metric('promotion_demotion', 'disparate_impact'):.4f}" + "}",
        "\\newcommand{\\QThreeSwapZemel}{" + f"{metric('promotion_demotion', 'zemel_proxy'):.4f}" + "}",
        "\\newcommand{\\QThreeNoGenderAcc}{" + f"{metric('no_gender', 'accuracy'):.4f}" + "}",
        "\\newcommand{\\QThreeNoGenderDi}{" + f"{metric('no_gender', 'disparate_impact'):.4f}" + "}",
        "\\newcommand{\\QThreeNoGenderZemel}{" + f"{metric('no_gender', 'zemel_proxy'):.4f}" + "}",
        "\\newcommand{\\QThreeReweighedAcc}{" + f"{metric('reweighed', 'accuracy'):.4f}" + "}",
        "\\newcommand{\\QThreeReweighedDi}{" + f"{metric('reweighed', 'disparate_impact'):.4f}" + "}",
        "\\newcommand{\\QThreeReweighedZemel}{" + f"{metric('reweighed', 'zemel_proxy'):.4f}" + "}",
        "\\newcommand{\\QThreeGroupThrAcc}{" + f"{metric('group_thresholds', 'accuracy'):.4f}" + "}",
        "\\newcommand{\\QThreeGroupThrDi}{" + f"{metric('group_thresholds', 'disparate_impact'):.4f}" + "}",
        "\\newcommand{\\QThreeGroupThrZemel}{" + f"{metric('group_thresholds', 'zemel_proxy'):.4f}" + "}",
        "\\newcommand{\\QThreeGroupThrZero}{" + f"{metric('group_thresholds', 'threshold_group_0'):.4f}" + "}",
        "\\newcommand{\\QThreeGroupThrOne}{" + f"{metric('group_thresholds', 'threshold_group_1'):.4f}" + "}",
    ]
    with macros_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(metrics_path), str(macros_path)


def main() -> None:
    args = parse_args()
    paths = get_paths()
    configure_seed(args.seed)

    payload: Dict = {
        "config": {
            "student_id": args.student_id,
            "model_index": args.model_index,
            "security_profile": args.security_profile,
            "swap_k": args.swap_k,
            "population_n": args.population_n,
            "unbounded_p": args.unbounded_p,
            "seed": args.seed,
            "download_mnist": args.download_mnist,
        },
        "figures": {},
    }

    security_error = None
    try:
        security, sec_figs = run_security(args, paths)
        payload["security"] = security
        payload["figures"].update(sec_figs)
    except Exception as exc:
        payload["security"] = {"error": str(exc)}
        security_error = str(exc)

    privacy, prv_figs = run_privacy(args, paths)
    payload["privacy"] = privacy
    payload["figures"].update(prv_figs)

    fairness, fair_figs = run_fairness(args, paths)
    payload["fairness"] = fairness
    payload["figures"].update(fair_figs)

    metrics_path, macros_path = write_results(paths, payload)

    print("Saved metrics JSON:", metrics_path)
    print("Saved TeX macros:", macros_path)
    for key, fig_path in payload["figures"].items():
        print(f"Saved {key}: {fig_path}")

    if security_error is not None:
        raise SystemExit(
            "Security pipeline failed. " + security_error + "\n"
            "Fairness/privacy artifacts were generated. "
            "Provide local MNIST files or allow download and rerun."
        )


if __name__ == "__main__":
    main()
