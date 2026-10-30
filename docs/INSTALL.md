# Installation

Use a separate Linux environment with Python 3.11 and an NVIDIA GPU for full
training. The method smoke check runs on CPU and does not need data or weights.

## Installation using published binary packages

The following combination has matching published PyTorch and MMCV packages.
It is an installation recipe, not a claim that the paper experiments were
rerun with this environment.

```bash
conda create -n freq-dino python=3.11 -y
conda activate freq-dino
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install mmcv==2.1.0 --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
python -m pip install -e . --no-build-isolation
python scripts/check_method.py
```

Run the commands from the repository root. The training launcher adds that root
to the import path for `my_custom_module3.py`. The editable package is the
bundled MMDetection framework; installing it alongside a different editable
MMDetection checkout in the same environment can select the wrong source.

Use the full `mmcv` package with compiled operators, not `mmcv-lite`. If a
matching wheel is unavailable for your OS, architecture, Python, or PyTorch
version, follow the official MMCV build instructions for that exact environment.
Do not mix a wheel built for PyTorch 2.1 with PyTorch 2.5.

Official references:

- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [MMCV 2.1 wheels for CUDA 12.1 / PyTorch 2.1](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html)
- [Building MMCV](https://mmcv.readthedocs.io/en/2.x/get_started/build.html)

## Locally checked environment

The release's CPU method checks and configuration checks were run in the
existing research environment:

| Component | Version |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| Transformers | 4.33.3 |
| NumPy | 1.26.4 |

The installed MMCV reports a CUDA 12.4 build. Its installation provenance is not
recorded, so this package does not claim there is a matching official wheel for
that exact local combination. The installation recipe above also pins OpenCV
4.11 instead of copying the local OpenCV 4.12 pin, whose declared NumPy
requirement conflicts with NumPy 1.26.

No new model training was performed while preparing this source release.

## Pretrained dependencies

The training configuration expects an official MMDetection-compatible
Grounding DINO Swin-B pretrained model and BERT-base-uncased. Obtain them from
their original providers and keep them outside version control.

- [MMDetection Grounding DINO model configuration and checkpoints](https://github.com/open-mmlab/mmdetection/tree/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/mm_grounding_dino)
- [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)

The default detector filename is
`grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth` under
`weights/checkpoint/`. You can instead pass `--pretrained /path/to/model.pth` to
the launcher. `--bert` accepts the local BERT directory or a model identifier.
BERT files can be fetched by Transformers when training starts unless you
provide a local directory. No pretrained or fine-tuned weights are included
in this source package.
