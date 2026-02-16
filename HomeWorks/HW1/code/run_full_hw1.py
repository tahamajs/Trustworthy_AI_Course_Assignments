"""
Full HW1 pipeline: run all assignment experiments (generalization + robustness),
then aggregate training_summary.csv, cross_domain_summary.csv, robustness_summary.csv
and copy figures to report/figures.
Use --demo for quick smoke runs; omit for full dataset runs.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'  copied {src.name} -> {dst}')
    else:
        print(f'  [WARN] missing {src}')


def parse_args():
    p = argparse.ArgumentParser(description='Run full HW1 experiment set and export report CSVs + figures.')
    p.add_argument('--demo', action='store_true', default=True, help='Use demo (small) data for fast runs')
    p.add_argument('--no-demo', action='store_false', dest='demo', help='Use full datasets')
    p.add_argument('--epochs', type=int, default=3, help='Epochs per run (demo); use 10+ for real')
    p.add_argument('--report-fig-dir', default='../report/figures')
    p.add_argument('--summary-dir', default='checkpoints/report_summary')
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    report_fig = (script_dir / args.report_fig_dir).resolve()
    summary_dir = (script_dir / args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_fig.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    demo = ['--demo'] if args.demo else []
    base_train = [py, 'train.py', '--epochs', str(args.epochs), '--batch-size', '128'] + demo

    # --- Training jobs ---
    jobs = [
        ('svhn_baseline', base_train + ['--dataset', 'svhn', '--save-dir', 'checkpoints/svhn_baseline']),
        ('svhn_no_bn', base_train + ['--dataset', 'svhn', '--use-bn', 'false', '--save-dir', 'checkpoints/svhn_no_bn']),
        ('svhn_label_smooth', base_train + ['--dataset', 'svhn', '--label-smoothing', '0.25', '--save-dir', 'checkpoints/svhn_label_smooth']),
        ('svhn_augment_digits_safe', base_train + ['--dataset', 'svhn', '--augment-preset', 'digits_safe', '--save-dir', 'checkpoints/svhn_augment_digits_safe']),
        ('svhn_pretrained', base_train + ['--dataset', 'svhn', '--pretrained', 'imagenet', '--save-dir', 'checkpoints/svhn_pretrained']),
        ('svhn_adam', base_train + ['--dataset', 'svhn', '--optimizer', 'adam', '--lr', '1e-3', '--save-dir', 'checkpoints/svhn_adam']),
        ('mnist_baseline', base_train + ['--dataset', 'mnist', '--save-dir', 'checkpoints/mnist_baseline']),
        ('cifar_baseline', base_train + ['--dataset', 'cifar10', '--cifar-split', '0.2', '--save-dir', 'checkpoints/cifar_baseline']),
        ('cifar_adv_fgsm', base_train + ['--dataset', 'cifar10', '--cifar-split', '0.2', '--adv-train', '--attack', 'fgsm', '--epsilon', '8/255', '--save-dir', 'checkpoints/cifar_adv_fgsm']),
        ('cifar_adv_pgd', base_train + ['--dataset', 'cifar10', '--cifar-split', '0.2', '--adv-train', '--attack', 'pgd', '--epsilon', '8/255', '--alpha', '2/255', '--iters', '7', '--save-dir', 'checkpoints/cifar_adv_pgd']),
        ('cifar_circle', base_train + ['--dataset', 'cifar10', '--cifar-split', '0.2', '--loss', 'circle', '--save-dir', 'checkpoints/cifar_circle']),
    ]

    for run_id, cmd in jobs:
        run_cmd(cmd, cwd=script_dir)

    # Fine-tune: MNIST -> 750 SVHN, freeze conv
    run_cmd(base_train + ['--dataset', 'mnist', '--save-dir', 'checkpoints/mnist_for_finetune'], cwd=script_dir)
    run_cmd(
        [py, 'train.py', '--finetune-from', 'checkpoints/mnist_for_finetune/best.pth', '--finetune-dataset', 'svhn',
         '--finetune-samples', '750', '--freeze-conv', '--epochs', '15', '--save-dir', 'checkpoints/mnist_finetune_svhn'] + demo,
        cwd=script_dir,
    )

    # --- Aggregate training_summary.csv from training_history.json ---
    training_rows = [('run_id', 'train_loss', 'train_acc', 'val_loss', 'val_acc')]
    for run_id, cmd in jobs:
        save_dir = script_dir / cmd[cmd.index('--save-dir') + 1]
        hist_path = save_dir / 'training_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
            if h.get('epoch'):
                i = -1
                training_rows.append((
                    run_id,
                    f"{h['train_loss'][i]:.4f}",
                    f"{h['train_acc'][i]:.2f}",
                    f"{h['val_loss'][i]:.4f}",
                    f"{h['val_acc'][i]:.2f}",
                ))
    # Add finetune run
    ft_hist = script_dir / 'checkpoints/mnist_finetune_svhn/training_history.json'
    if ft_hist.exists():
        with open(ft_hist) as f:
            h = json.load(f)
        if h.get('epoch'):
            i = -1
            training_rows.append(('mnist_finetune_svhn', f"{h['train_loss'][i]:.4f}", f"{h['train_acc'][i]:.2f}", f"{h['val_loss'][i]:.4f}", f"{h['val_acc'][i]:.2f}"))

    with open(summary_dir / 'training_summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(training_rows)
    print('[OK] Wrote', summary_dir / 'training_summary.csv')

    # --- Cross-domain evaluation and cross_domain_summary.csv ---
    cross_rows = [('source_model', 'eval_dataset', 'acc')]
    gen_evals = [
        ('checkpoints/svhn_baseline/best.pth', ['svhn', 'mnist'], None),
        ('checkpoints/svhn_no_bn/best.pth', ['svhn', 'mnist'], None),
        ('checkpoints/svhn_label_smooth/best.pth', ['svhn', 'mnist'], None),
        ('checkpoints/svhn_adam/best.pth', ['svhn', 'mnist'], None),
        ('checkpoints/svhn_pretrained/best.pth', ['svhn', 'mnist'], '--pretrained'),
        ('checkpoints/mnist_baseline/best.pth', ['mnist', 'svhn'], None),
        ('checkpoints/mnist_finetune_svhn/best.pth', ['svhn', 'mnist'], None),
    ]
    for ckpt_rel, datasets, pretrained_flag in gen_evals:
        ckpt = script_dir / ckpt_rel
        if not ckpt.exists():
            continue
        run_cmd(
            [py, 'eval.py', '--checkpoint', str(ckpt), '--eval-datasets', ','.join(datasets),
             '--cross-eval-csv', str(ckpt) + '.cross_eval.csv', '--dataset', datasets[0]] + (['--pretrained'] if pretrained_flag else []) + demo,
            cwd=script_dir,
        )
        csv_path = Path(str(ckpt) + '.cross_eval.csv')
        if csv_path.exists():
            with open(csv_path) as f:
                r = csv.DictReader(f)
                for row in r:
                    cross_rows.append((Path(ckpt_rel).parent.name, row['eval_dataset'], row['accuracy']))

    with open(summary_dir / 'cross_domain_summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(cross_rows)
    print('[OK] Wrote', summary_dir / 'cross_domain_summary.csv')

    # --- Robustness: eval clean + FGSM (epsilon 0.1 for grid, sweep for table), write robustness_summary.csv ---
    robust_rows = [('model', 'clean', 'fgsm_01', 'fgsm_8_255', 'pgd', 'noise')]
    for run_id, ckpt_subpath in [
        ('cifar_baseline', 'checkpoints/cifar_baseline/best.pth'),
        ('cifar_adv_fgsm', 'checkpoints/cifar_adv_fgsm/best.pth'),
        ('cifar_adv_pgd', 'checkpoints/cifar_adv_pgd/best.pth'),
        ('cifar_circle', 'checkpoints/cifar_circle/best.pth'),
    ]:
        ckpt = script_dir / ckpt_subpath
        if not ckpt.exists():
            continue
        run_cmd(
            [py, 'eval.py', '--dataset', 'cifar10', '--checkpoint', str(ckpt),
             '--umap', '--umap-path', str(ckpt.parent / 'umap.png'),
             '--save-grid', '--grid-path', str(ckpt.parent / 'grid_eps01.png'), '--epsilon', '0.1'] + demo,
            cwd=script_dir,
        )
        # Collect metrics from a short eval (we need evaluate_accuracy_under_attack for each attack type)
        run_cmd(
            [py, 'eval.py', '--dataset', 'cifar10', '--checkpoint', str(ckpt),
             '--save-attack-sweep', '--attack-sweep-path', str(ckpt.parent / 'robustness_sweep.png'),
             '--sweep-epsilons', '0/255,2/255,4/255,8/255,0.1', '--sweep-iters', '3', '--sweep-max-batches', '4',
             '--metrics-path', str(ckpt.parent / 'robust_metrics.json')] + demo,
            cwd=script_dir,
        )
        metrics_file = ckpt.parent / 'robust_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
            sweep = m.get('robustness_sweep', {})
            fgsm_acc = sweep.get('fgsm_acc') or []
            pgd_acc = sweep.get('pgd_acc') or []
            noise_acc = sweep.get('noise_acc') or []
            # Last epsilon in sweep is 0.1
            fgsm_01 = f'{fgsm_acc[-1]:.2f}' if len(fgsm_acc) >= 5 else '-'
            fgsm_8 = f'{fgsm_acc[3]:.2f}' if len(fgsm_acc) > 3 else (f'{fgsm_acc[-1]:.2f}' if fgsm_acc else '-')
            robust_rows.append((
                run_id,
                f"{sweep.get('clean_acc', 0):.2f}",
                fgsm_01,
                fgsm_8,
                f'{pgd_acc[-1]:.2f}' if pgd_acc else '-',
                f'{noise_acc[-1]:.2f}' if noise_acc else '-',
            ))

    with open(summary_dir / 'robustness_summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(robust_rows)
    print('[OK] Wrote', summary_dir / 'robustness_summary.csv')

    # --- Copy figures to report/figures ---
    copies = [
        ('checkpoints/svhn_baseline/training_curves.png', 'training_curves.png'),
        ('checkpoints/svhn_baseline/best.pth.umap.png', 'umap_svhn.png'),
        ('checkpoints/svhn_baseline/best.pth.grid.png', 'adv_examples.png'),
        ('checkpoints/cifar_baseline/umap.png', 'umap_cifar_baseline.png'),
        ('checkpoints/cifar_baseline/grid_eps01.png', 'adv_examples_eps01.png'),
        ('checkpoints/cifar_baseline/robustness_sweep.png', 'robustness_sweep.png'),
    ]
    for src_rel, dst_name in copies:
        copy_if_exists(script_dir / src_rel, report_fig / dst_name)

    print('[DONE] Summary CSVs in', summary_dir, '; figures in', report_fig)


if __name__ == '__main__':
    main()
