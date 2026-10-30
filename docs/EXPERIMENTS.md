# Method and training procedure

The release covers the DSA Grounding DINO baseline, Freq-DINO, and its two branch/fusion ablations. It contains code and configurations; pretrained checkpoints, trained weights, training logs, private data, and experimental result files are not bundled.

## Methods

| `--method` | Configuration in `configs/mm_grounding_dino/` | Change from the baseline |
| --- | --- | --- |
| `baseline` | `dsa_fewshot_base.py` | Grounding DINO Swin-B adaptation with no prior branch |
| `freq_prior` | `dsa_fewshot_freq_prior.py` | Fourier residual prior and learned spatial gated residual fusion |
| `no_gate` | `dsa_fewshot_freq_no_gate.py` | Same Fourier branch; residual addition without the learned spatial gate |
| `gray_branch_train` | `dsa_fewshot_gray_branch_train.py` | Normalized grayscale prior with the same lightweight encoder and gated fusion |

The inherited `dsa_fewshot_gray_branch.py` supplies the grayscale model definition. Its `*_train.py` wrapper disables optional image/statistics hooks, matching the repeated ablation runs. Required upstream base configurations are included only to resolve these methods.

`GroundingDINOLoRA` is the existing custom class name. Here `lora_r=0`: this protocol does not train LoRA adapters. The entire Swin-B visual backbone and BERT text encoder are frozen. Detection adaptation modules are trainable; the prior encoder and fusion modules are also trainable when enabled. `freq_num_peaks=3` denotes three independent Fourier peak pairs, with each peak suppressed together with its conjugate counterpart.

## Local inputs

Prepare existing support and validation files as described in [DATA.md](DATA.md). Obtain the original MM Grounding DINO Swin-B pretrained checkpoint separately and keep it locally under:

```text
weights/checkpoint/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
```

No user-trained weights are needed to start this fine-tuning procedure, and none are included in the release. The default BERT identifier is `bert-base-uncased`; a local Hugging Face-format BERT directory can be supplied for offline use. Model construction may download BERT when using the default identifier and it is not already cached.

The primary configuration reads these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DSA_DATA_ROOT` | `data/dsa` | Image and annotation root |
| `FREQ_DINO_WEIGHTS_DIR` | `weights/checkpoint` | Directory containing the pretrained Grounding DINO checkpoint |
| `FREQ_DINO_BERT` | `bert-base-uncased` | BERT model identifier or local model directory |

The launcher provides corresponding `--data-root`, `--weights-dir`, and `--bert` options. `--pretrained` can point directly to the initial checkpoint. Explicit relative CLI paths are resolved against the shell's current directory; default repository paths are anchored to the repository. The launcher executes training from the repository root and adds it to `PYTHONPATH` for `my_custom_module3` registration.

## Single experiment

First inspect the command without importing MMDetection, creating outputs, or downloading models:

```bash
python scripts/run_experiment.py train \
  --method freq_prior --shot 10 --split-seed 1 --train-seed 42 --dry-run
```

Run training after installing dependencies and placing the local inputs:

```bash
python scripts/run_experiment.py train \
  --method freq_prior --shot 10 --split-seed 1 --train-seed 42 --gpu 0
```

Choose `baseline`, `no_gate`, or `gray_branch_train` for the corresponding variant. `--split-seed` selects files; it does not resample data. `--train-seed` controls MMEngine's training randomness and defaults to 42 regardless of support-set seed. Use the same shot and split seed for paired method runs. Default outputs go to `work_dirs/<method>_<shot>shot_seed<seed>`; pass a distinct `--work-dir` for a repeat rather than mixing its logs with an existing run. The launcher does not automatically resume training.

The launcher verifies that the annotation files, image directory, label map, and initial checkpoint exist before starting. Detailed annotation verification is a separate `prepare_data.py verify` step. `--help` and `--dry-run` work with Python alone. An actual training run uses the installed MMDetection/MMCV/PyTorch stack and a suitable GPU; CPU-only config checks do not establish training performance or numerical reproduction.

## Preserved training settings

| Setting | Value |
| --- | --- |
| Epochs | 120 |
| Training batch size | 4 per process |
| Training RNG seed | 42; `deterministic=False` |
| Validation | Every 10 epochs, on the remaining images of that support split |
| Optimizer | AdamW; learning rate `5e-5`, weight decay `1e-4` |
| Gradient clipping | Maximum norm 0.1, norm type 2 |
| Schedule | 80-iteration linear warmup together with epoch-based cosine decay to `1e-6`; both begin at step/epoch 0 |
| Branch | 256 channels, 4 levels, 3 independent peak pairs |
| Frequency settings | Center radius 30, peak radius 8, percentile 0.5, smooth sigma 20, border ratio 0.03 |
| Training inputs | Concatenated detection and visual-grounding views of the same support images |

The multiscale resize, flip, optional crop, text processing, and validation transforms are preserved in the source configurations. Optional prior visualizations and fusion statistics for `freq_prior`/`no_gate` are saved within each launcher work directory. They are locally generated diagnostics, not files distributed with this release.

## AP checkpoint saving versus AP50 table selection

The historical training configuration saves its best checkpoint according to **AP**, using `save_best='coco/bbox_mAP'`. AP averages IoU thresholds 0.50–0.95. This is retained for compatibility with the original training process.

The paper's original-batch tables instead choose the validation-log record with the largest **AP50** (`coco/bbox_mAP_50`), using AP to break ties, and report AP, AP50, and AP75 from that same record. Exact remaining ties retain the first matching log record. Those are validation-selected results on the same split, not an independent held-out test estimate.

Consequently, a `best_coco_bbox_mAP_*.pth` file is not necessarily the AP50-selected epoch. The original checkpoint hook retains only two interval checkpoints, so the AP50-selected epoch may no longer have a saved weight file even when its validation metrics remain in the log. Do not silently use the AP-best checkpoint as if it were the AP50-selected checkpoint. If a new run needs every validation epoch's weights, explicitly retain the interval checkpoints when configuring that run; this changes storage retention, not the model or optimizer protocol.

This scoped release provides training with periodic validation. It does not package independent evaluation, cross-batch inference, detector-comparison experiments, or trained checkpoints.
