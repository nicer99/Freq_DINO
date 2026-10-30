#!/usr/bin/env python3
"""Launch one DSA method training run, including its periodic validation.

The CLI and --dry-run use only the Python standard library. No model is built
and no weights are downloaded until the generated training command is executed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = 'grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth'
CONFIGS = {
    'baseline': 'dsa_fewshot_base.py',
    'freq_prior': 'dsa_fewshot_freq_prior.py',
    'no_gate': 'dsa_fewshot_freq_no_gate.py',
    'gray_branch_train': 'dsa_fewshot_gray_branch_train.py',
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest='action', required=True)
    train = actions.add_parser('train', help='train one method; validate every 10 epochs')
    train.add_argument('--method', choices=CONFIGS, default='freq_prior')
    train.add_argument('--shot', choices=(1, 3, 5, 10), type=int, default=10)
    train.add_argument('--split-seed', type=positive_int, default=1,
                       help='existing support-set split ID, not the optimizer RNG seed')
    train.add_argument('--train-seed', type=int, default=42,
                       help='MMEngine training RNG seed; paper default: 42')
    train.add_argument('--data-root', default=os.environ.get('DSA_DATA_ROOT', str(ROOT / 'data/dsa')))
    train.add_argument('--weights-dir', default=os.environ.get('FREQ_DINO_WEIGHTS_DIR', str(ROOT / 'weights/checkpoint')))
    train.add_argument('--bert', default=os.environ.get('FREQ_DINO_BERT', 'bert-base-uncased'),
                       help='Hugging Face model name or an existing local BERT directory')
    train.add_argument('--pretrained', help='initial Grounding DINO checkpoint; overrides --weights-dir')
    train.add_argument('--work-dir', help='output directory (default: work_dirs/METHOD_SHOTshot_seedSEED)')
    train.add_argument('--gpu', help='single CUDA device exposed to this run, e.g. 0')
    train.add_argument('--dry-run', action='store_true',
                       help='print command and effective paths without requiring data, MMDetection, or models')
    return result


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str], list[Path]]:
    data_root = absolute_path(args.data_root)
    weights_dir = absolute_path(args.weights_dir)
    pretrained = absolute_path(args.pretrained) if args.pretrained else weights_dir / CHECKPOINT_NAME
    work_dir = absolute_path(args.work_dir) if args.work_dir else (
        ROOT / 'work_dirs' / f'{args.method}_{args.shot}shot_seed{args.split_seed}')
    bert = args.bert
    if Path(bert).expanduser().is_dir() or bert.startswith(('.', '/', '~')):
        bert = str(absolute_path(bert))
    env = os.environ.copy()
    env.update(DSA_DATA_ROOT=str(data_root),
               FREQ_DINO_WEIGHTS_DIR=str(weights_dir), FREQ_DINO_BERT=bert)
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_ann = f'few_shot_dataset/train_{args.shot}shot_seed{args.split_seed}.jsonl'
    val_ann = f'few_shot_dataset/val_{args.shot}shot_seed{args.split_seed}.json'
    config = ROOT / 'configs/mm_grounding_dino' / CONFIGS[args.method]
    options = [
        f'train_dataloader.dataset.datasets.0.ann_file={train_ann}',
        f'train_dataloader.dataset.datasets.1.ann_file={train_ann}',
        f'val_dataloader.dataset.ann_file={val_ann}',
        f'test_dataloader.dataset.ann_file={val_ann}',
        f'val_evaluator.ann_file={data_root / val_ann}',
        f'test_evaluator.ann_file={data_root / val_ann}',
        f'randomness.seed={args.train_seed}',
        f'load_from={pretrained}',
    ]
    if args.method in ('freq_prior', 'no_gate'):
        # Keep diagnostics within this run rather than sharing output paths.
        options += [f'custom_hooks.1.save_dir={work_dir / "freq_prior_vis"}',
                    f'custom_hooks.2.save_dir={work_dir / "freq_prior_stats"}']
    command = [sys.executable, str(ROOT / 'tools/train.py'), str(config),
               '--work-dir', str(work_dir), '--cfg-options', *options]
    required = [data_root / 'images', data_root / 'label_map.json',
                data_root / train_ann, data_root / val_ann, pretrained]
    return command, env, required


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    command, env, required = build_command(args)
    print(f'Working directory: {ROOT}')
    for name in ('DSA_DATA_ROOT', 'FREQ_DINO_WEIGHTS_DIR', 'FREQ_DINO_BERT'):
        print(f'{name}={env[name]}')
    if args.gpu is not None:
        print(f'CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}')
    print('Training: 120 epochs; validation every 10 epochs; checkpoint best = AP.')
    print('Paper table selection: maximum validation AP50, then AP; see docs/EXPERIMENTS.md.')
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        cli.error('missing training inputs:\n  ' + '\n  '.join(missing) +
                  '\nSee docs/DATA.md and docs/EXPERIMENTS.md. No training was started.')
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
