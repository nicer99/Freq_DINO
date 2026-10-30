# Freq-DINO

Method implementation and training code for **Freq-DINO: Frequency-Guided
Vision-Language Detection for Few-Shot Defect Inspection of Dense Lithographic
Patterns**.

Freq-DINO adds an FFT-derived structural prior and gated multi-scale residual
fusion to Grounding DINO. The training configurations use a frozen Swin-B
backbone and BERT encoder while adapting the detector and the new frequency
branch. The implementation uses three conjugate peak pairs (`K=3`) and no LoRA.

This repository contains source code, configurations, and training
instructions. It does not distribute datasets, pretrained models, authors'
trained weights, training logs, prediction images, or experimental result files.

## Contents

| Path | Purpose |
|---|---|
| `my_custom_module3.py` | Frequency prior, trainable encoder, gated fusion, and detector integration |
| `configs/mm_grounding_dino/dsa_*.py` | Freq-DINO, baseline, and training controls |
| `scripts/run_experiment.py` | Portable training launcher with shot and support-seed options |
| `scripts/prepare_data.py` | Data verification, annotation conversion, and split preparation |
| `scripts/check_method.py` | CPU method check using synthetic tensors |
| `mmdet/`, `tools/train.py` | Bundled MMDetection framework and training entry point |
| `docs/` | Installation, method, data format, and training details |

## Start here

1. Follow [installation instructions](docs/INSTALL.md) in a separate environment.
2. Prepare your data using [the data format and split instructions](docs/DATA.md).
3. Obtain the official pretrained dependencies from their original providers;
   see [pretrained dependencies](docs/INSTALL.md#pretrained-dependencies).
4. Inspect the training command, then run it:

```bash
python scripts/run_experiment.py train --method freq_prior --shot 10 --split-seed 1 --dry-run
python scripts/run_experiment.py train --method freq_prior --shot 10 --split-seed 1
```

The dry run prints the command and does not load weights or start training.
Supply `--data-root`, `--pretrained`, and `--bert` when your local paths differ
from the defaults. See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full workflow,
baseline and control commands, and training checkpoint selection.

## Method and training protocol

The prior suppresses dominant periodic peaks, extracts residual structural
cues, and encodes them into four feature scales. Learned gates control their
addition to visual features. [METHOD.md](docs/METHOD.md) explains the exact FFT
operations, parameter meanings, and frozen/trainable modules.

The main configurations use 120 epochs, batch size 4, AdamW with learning rate
`5e-5` and weight decay `1e-4`, 80 warm-up iterations, a cosine schedule,
gradient clipping at 0.1, training seed 42, and validation every 10 epochs.
Support-set seeds identify different data selections and are separate from the
training seed.

Each support set has OD and VG supervision views sharing the same images and
boxes. The paper uses existing 1/3/5/10-shot splits; obtaining the historical
splits is necessary to reproduce the same selected images. Newly generated
splits must not be presented as exact copies of the historical seeds 1–3.

Validation is part of the training procedure. The default saved best checkpoint
uses AP, while the original-batch paper tables selected epochs by validation
AP50 with AP as a tie-breaker. These are different selection rules. The source
release does not include any selected checkpoint or claim a new reproduction
of the paper's numerical results.

## Availability and licensing

The datasets analyzed in the study are available from the corresponding
authors on reasonable request. They are not bundled in this repository.

The authors have not selected a license for original Freq-DINO code. Bundled
MMDetection code retains its Apache-2.0 license and notices; see
[LICENSE.md](LICENSE.md) and [UPSTREAM.md](docs/UPSTREAM.md). The presence of
upstream license text is not a repository-wide license declaration.

中文上传说明见 [UPLOAD_ZH.md](UPLOAD_ZH.md)。
