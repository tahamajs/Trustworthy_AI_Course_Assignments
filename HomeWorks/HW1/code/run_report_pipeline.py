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
    ]
    if not args.full_run:
        eval_cmd.append('--demo')
    run_step(eval_cmd, cwd=script_dir)

    copy_if_exists(save_dir / 'training_curves.png', report_fig_dir / 'training_curves.png')
    copy_if_exists(umap_path, report_fig_dir / 'umap_features.png')
    copy_if_exists(grid_path, report_fig_dir / 'adv_examples.png')

    print('[DONE] Report artifacts updated in:', report_fig_dir)


if __name__ == '__main__':
    main()
