"""Generate report-ready plots from saved benchmark artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils


def _default_figures_dir() -> Path:
    # q5_codes -> code -> HW3/report/figures
    return Path(__file__).resolve().parents[2] / "report" / "figures"


def _load_array(path: Path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def export_report_plots(dataset, trainer, model_type, epsilon, seed, lambd, output_dir=None):
    """Export report figures if the required result artifacts are available."""
    metrics_base = Path(utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, epsilon, seed))
    acc = _load_array(Path(str(metrics_base) + "_accs.npy"))
    mcc = _load_array(Path(str(metrics_base) + "_mccs.npy"))
    valid = _load_array(Path(str(metrics_base) + "_valid.npy"))
    cost = _load_array(Path(str(metrics_base) + "_cost.npy"))

    if any(arr is None for arr in [acc, mcc, valid, cost]):
        missing = [
            suffix
            for suffix, arr in [
                ("_accs.npy", acc),
                ("_mccs.npy", mcc),
                ("_valid.npy", valid),
                ("_cost.npy", cost),
            ]
            if arr is None
        ]
        print(f"Skipping plot export. Missing artifacts for {metrics_base}: {', '.join(missing)}")
        return

    output_dir = _default_figures_dir() if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_value = float(np.asarray(acc).reshape(-1)[-1])
    mcc_value = float(np.asarray(mcc).reshape(-1)[-1])
    valid = np.asarray(valid).astype(bool).reshape(-1)
    cost = np.asarray(cost).reshape(-1)

    # Figure 1: per-individual recourse costs (expected by the report template).
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x_idx = np.arange(cost.shape[0])
    colors = np.where(valid, "#2A9D8F", "#E76F51")
    ax.bar(x_idx, cost, color=colors, edgecolor="#1f2933")
    if valid.any():
        mean_cost = float(np.mean(cost[valid]))
        ax.axhline(mean_cost, linestyle="--", linewidth=1.5, color="#264653", label=f"Mean valid cost: {mean_cost:.3f}")
        ax.legend(frameon=False)
    ax.set_title(f"Recourse Cost per Explained Instance ({dataset}, {model_type}, {trainer})")
    ax.set_xlabel("Explained Instance Index")
    ax.set_ylabel("L1 Recourse Cost")
    ax.set_xticks(x_idx)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "recourse_costs.png", dpi=220)
    plt.close(fig)

    # Figure 2: summary metrics for quick inclusion in the report.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(["Accuracy", "MCC"], [acc_value, mcc_value], color=["#2A9D8F", "#264653"], edgecolor="#1f2933")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Classifier Metrics")
    axes[0].grid(axis="y", alpha=0.25)

    valid_count = int(valid.sum())
    invalid_count = int(valid.shape[0] - valid_count)
    axes[1].bar(["Valid", "Invalid"], [valid_count, invalid_count], color=["#2A9D8F", "#E76F51"], edgecolor="#1f2933")
    axes[1].set_title("Recourse Validity Count")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"Benchmark Summary ({dataset}, {model_type}, {trainer}, seed={seed})")
    fig.tight_layout()
    fig.savefig(output_dir / "classifier_recourse_summary.png", dpi=220)
    plt.close(fig)

    print(f"Saved plots to {output_dir}")

