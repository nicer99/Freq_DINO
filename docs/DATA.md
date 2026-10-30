# Data preparation

This repository contains preparation code, not the DSA images, annotations, or historical support-set files. The datasets analyzed in the study are available from the corresponding author on reasonable request. Request the original split files when reproducing the paper; independently resampling the data does not reproduce a historical partition.

## Directory and labels

Set `DSA_DATA_ROOT` or pass `--data-root` to the scripts. The default is `data/dsa` under the repository root.

```text
data/dsa/
├── images/                         # files referenced by annotation filenames
├── label_map.json
├── custom_coco_dataset.json        # needed for split generation / COCO conversion
└── few_shot_dataset/
    ├── train_10shot_seed1.jsonl     # ODVG support annotations
    ├── train_10shot_seed1_coco.json # optional COCO representation
    └── val_10shot_seed1.json        # COCO validation annotations
```

The preparation code expects this exact mapping:

| Category ID | Class name |
| --- | --- |
| 0 | `Missing_Hole` |
| 1 | `Connected_hole` |
| 2 | `Block` |
| 3 | `Particle` |
| 4 | `Dislocation` |

`label_map.json` maps string IDs to names:

```json
{"0": "Missing_Hole", "1": "Connected_hole", "2": "Block", "3": "Particle", "4": "Dislocation"}
```

COCO boxes use `[x, y, width, height]`. The ODVG training file has one JSON object per image, with `filename`, `height`, `width`, `detection.instances`, and `grounding`. ODVG boxes use `[x1, y1, x2, y2]`. Each detection instance has `bbox`, `label`, and `category`; each grounding region has `bbox`, `phrase`, and character-offset `tokens_positive` referring to `grounding.caption`. The same support file is read in detection and grounding modes by the two training datasets.

## Use existing partitions

For a paper experiment, place its original files in the tree above. Do not regenerate a partition simply because its seed number is known.

```bash
python scripts/prepare_data.py verify \
  --data-root data/dsa --shots 1 3 5 10 --seeds 1 2 3 --check-images
```

For the main ten-seed comparison, also verify seeds 4–10 after obtaining those files. `verify` checks the class mapping, annotation structure, box extents, grounding span bounds, image sizes, duplicate filenames, and support/validation filename overlap. `--check-images` additionally checks file presence; it does not decode images or establish their scientific validity. It prints actual support-instance counts because retaining every annotation on selected images can exceed the nominal budget.

## Generate additional splits

The included sampling function is the original extension used for seeds 4–10. It prefers images with a single annotated instance, falls back to other images if necessary, keeps all annotations on selected images, adds two background images, and assigns the remaining images to validation. Its local RNG is `random.Random(seed * 1009 + shot * 9173)`; the full COCO image and annotation ordering is part of the input.

```bash
python scripts/prepare_data.py splits \
  --data-root data/dsa --shots 1 3 5 10 --seeds 4 5 6 7 8 9 10
```

This writes support ODVG, support COCO, and validation COCO files. It creates the standard `label_map.json` when absent. Existing complete split files are skipped; a partial set is rejected to avoid combining a generated partition with historical files. `--overwrite` explicitly replaces generated annotations.

**This generator cannot reconstruct the original seeds-1–3 files.** It refuses those IDs. Exact reproduction of later partitions also requires the same source annotations and ordering. The nominal shot budget concerns per-class support instances, but selection occurs at image level and includes every annotation on a selected image; inspect the reported actual counts.

The historical caption recipe is retained verbatim. It uses class-specific DSA descriptions and a single image caption assigned while iterating over annotations; it is not a general-purpose caption generator for arbitrary multi-class scenes. A different dataset or caption scheme requires explicit adaptation and is a different experiment.

## Convert an existing support file without resampling

The optional conversion command preserves the selected images and converts detection boxes to COCO. The four method configurations train directly from ODVG, so this step is not required for them.

```bash
python scripts/prepare_data.py convert \
  --data-root data/dsa --shots 10 --seeds 1 2 3
```

It reads the class definitions from `custom_coco_dataset.json` and writes `train_<shot>shot_seed<seed>_coco.json`. Existing outputs are skipped unless `--overwrite` is supplied. None of these commands uploads data or contacts an external service.
