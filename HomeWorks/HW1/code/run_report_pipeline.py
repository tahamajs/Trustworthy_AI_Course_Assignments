import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(cmd, cwd):
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'[OK] Copied {src} -> {dst}')
    else:
        print(f'[WARN] Missing artifact: {src}')


def parse_args():
    p = argparse.ArgumentParser(description='Run a complete HW1 demo pipeline and export report figures.')
    p.add_argument('--dataset', default='svhn', choices=['svhn', 'mnist', 'cifar10'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    p.add_argument('--attack', default='fgsm', choices=['fgsm', 'pgd'])
    p.add_argument('--epsilon', default='8/255')
    p.add_argument('--alpha', default='2/255')
    p.add_argument('--iters', type=int, default=7)
    p.add_argument('--save-dir', default='checkpoints/svhn_demo')
    p.add_argument('--report-fig-dir', default='../report/figures')
    p.add_argument('--grid-samples', type=int, default=8)
    p.add_argument('--sweep-epsilons', default='0/255,2/255,4/255,8/255,12/255')
    p.add_argument('--sweep-iters', type=int, default=3)
    p.add_argument('--sweep-max-batches', type=int, default=4)
    p.add_argument('--coverage-points', type=int, default=20)
    p.add_argument('--topk-list', default='1,2,3,5')
    p.add_argument('--pgd-iters-list', default='1,2,4,7,10')
    p.add_argument('--iter-sweep-max-batches', type=int, default=4)
    p.add_argument('--full-run', action='store_true', help='Disable demo mode and run on full dataset.')
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    save_dir = (script_dir / args.save_dir).resolve()
    report_fig_dir = (script_dir / args.report_fig_dir).resolve()
    python_bin = sys.executable

    train_cmd = [
        python_bin,
        'train.py',
        '--dataset',
        args.dataset,
        '--epochs',
        str(args.epochs),
        '--batch-size',
        str(args.batch_size),
        '--optimizer',
        args.optimizer,
        '--save-dir',
        str(save_dir),
    ]
    if not args.full_run:
        train_cmd.append('--demo')
    run_step(train_cmd, cwd=script_dir)

    ckpt_path = save_dir / 'best.pth'
    umap_path = save_dir / 'best.pth.umap.png'
    grid_path = save_dir / 'best.pth.grid.png'
    confusion_path = save_dir / 'best.pth.confusion.png'
    per_class_path = save_dir / 'best.pth.per_class.png'
    prf1_path = save_dir / 'best.pth.prf1.png'
    calibration_path = save_dir / 'best.pth.calibration.png'
    confidence_coverage_path = save_dir / 'best.pth.confidence_coverage.png'
    topk_path = save_dir / 'best.pth.topk.png'
    sweep_path = save_dir / 'best.pth.robustness_sweep.png'
    pgd_iter_sweep_path = save_dir / 'best.pth.pgd_iter_sweep.png'
    metrics_path = save_dir / 'best.pth.metrics.json'

    eval_cmd = [
        python_bin,
        'eval.py',
        '--dataset',
        args.dataset,
        '--checkpoint',
        str(ckpt_path),
        '--umap',
        '--save-grid',
        '--attack',
        args.attack,
        '--epsilon',
        args.epsilon,
        '--alpha',
        args.alpha,
        '--iters',
        str(args.iters),
        '--grid-samples',
        str(args.grid_samples),
        '--umap-path',
        str(umap_path),
        '--grid-path',
        str(grid_path),
        '--save-confusion',
        '--confusion-path',
        str(confusion_path),
        '--save-per-class',
        '--per-class-path',
        str(per_class_path),
        '--save-prf1',
        '--prf1-path',
        str(prf1_path),
        '--save-calibration',
        '--calibration-path',
        str(calibration_path),
        '--save-confidence-coverage',
        '--confidence-coverage-path',
        str(confidence_coverage_path),
        '--coverage-points',
        str(args.coverage_points),
        '--save-topk',
        '--topk-path',
        str(topk_path),
        '--topk-list',
        args.topk_list,
        '--save-attack-sweep',
        '--attack-sweep-path',
        str(sweep_path),
        '--sweep-epsilons',
        args.sweep_epsilons,
        '--sweep-iters',
        str(args.sweep_iters),
        '--sweep-max-batches',
        str(args.sweep_max_batches),
        '--save-pgd-iter-sweep',
        '--pgd-iter-sweep-path',
        str(pgd_iter_sweep_path),
        '--pgd-iters-list',
        args.pgd_iters_list,
        '--iter-sweep-max-batches',
        str(args.iter_sweep_max_batches),
        '--metrics-path',
        str(metrics_path),
    ]
    if not args.full_run:
        eval_cmd.append('--demo')
    run_step(eval_cmd, cwd=script_dir)

    copy_if_exists(save_dir / 'training_curves.png', report_fig_dir / 'training_curves.png')
    copy_if_exists(umap_path, report_fig_dir / 'umap_features.png')
    copy_if_exists(grid_path, report_fig_dir / 'adv_examples.png')
    copy_if_exists(confusion_path, report_fig_dir / 'confusion_matrix.png')
    copy_if_exists(per_class_path, report_fig_dir / 'per_class_accuracy.png')
    copy_if_exists(prf1_path, report_fig_dir / 'classwise_prf1.png')
    copy_if_exists(calibration_path, report_fig_dir / 'reliability_diagram.png')
    copy_if_exists(confidence_coverage_path, report_fig_dir / 'confidence_coverage.png')
    copy_if_exists(topk_path, report_fig_dir / 'topk_accuracy.png')
    copy_if_exists(sweep_path, report_fig_dir / 'robustness_sweep.png')
    copy_if_exists(pgd_iter_sweep_path, report_fig_dir / 'pgd_iter_sweep.png')
    copy_if_exists(metrics_path, report_fig_dir / 'metrics_summary.json')

    print('[DONE] Report artifacts updated in:', report_fig_dir)


if __name__ == '__main__':
    main()
