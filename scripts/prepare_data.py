#!/usr/bin/env python3
"""Prepare privately obtained DSA data without bundling any source images.

The sampling and caption functions retain the original seeds-4--10 recipe.
This generator does not reconstruct the historical seeds-1--3 partitions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random

CLASSES = ('Missing_Hole', 'Connected_hole', 'Block', 'Particle', 'Dislocation')
REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_CAPTIONS = {
    0: (
        "A DSA hexagonally packed contact hole array exhibits a missing hole "
        "defect, where one or more contact holes are absent or significantly "
        "reduced in size.",
        "missing hole defect",
        [[55, 74]],
    ),
    1: (
        "A DSA hexagonally packed contact hole array shows a connected hole "
        "defect, where multiple adjacent contact holes merge together.",
        "connected hole defect",
        [[52, 73]],
    ),
    2: (
        "A DSA hexagonally packed contact hole array exhibits a block defect, "
        "which typically appears as a blocky dark region.",
        "block defect",
        [[55, 67]],
    ),
    3: (
        "A DSA hexagonally packed contact hole array shows a Particle defect, "
        "appearing as a localized bright spot caused by foreign contamination.",
        "Particle defect",
        [[52, 67]],
    ),
    4: (
        "A DSA hexagonally packed contact hole array exhibits a dislocation "
        "defect that disrupts the regular hexagonal packing of the holes.",
        "dislocation defect",
        [[55, 73]],
    ),
}
BACKGROUND_CAPTION = (
    "A DSA hexagonally packed contact hole array shows no visible defects.")


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reindex_coco(coco: dict, image_ids: list[int]) -> dict:
    id_to_image = {int(image["id"]): image for image in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    images = []
    annotations = []
    ann_id = 0
    for new_id, old_id in enumerate(image_ids):
        image = dict(id_to_image[old_id])
        image["id"] = new_id
        images.append(image)
        for ann in anns_by_image.get(old_id, []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = new_id
            annotations.append(new_ann)
            ann_id += 1

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": images,
        "annotations": annotations,
    }


def coco_to_jsonl_records(coco_split: dict) -> list[dict]:
    categories = {int(cat["id"]): cat["name"] for cat in coco_split["categories"]}
    anns_by_image = defaultdict(list)
    for ann in coco_split["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    records = []
    for image in coco_split["images"]:
        image_id = int(image["id"])
        instances = []
        regions = []
        caption = BACKGROUND_CAPTION
        for ann in anns_by_image.get(image_id, []):
            cid = int(ann["category_id"])
            x, y, w, h = ann["bbox"]
            bbox = [x, y, x + w, y + h]
            name = categories[cid]
            instances.append({
                "bbox": bbox,
                "label": cid,
                "category": name,
            })
            caption, phrase, tokens_positive = CLASS_CAPTIONS[cid]
            regions.append({
                "bbox": bbox,
                "phrase": phrase,
                "tokens_positive": tokens_positive,
            })

        records.append({
            "filename": image["file_name"],
            "height": int(image["height"]),
            "width": int(image["width"]),
            "detection": {
                "instances": instances,
            },
            "grounding": {
                "caption": caption,
                "regions": regions,
            },
        })
    return records


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_split(coco: dict, shot: int, seed: int,
                 background_images: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed * 1009 + shot * 9173)
    anns_by_class = defaultdict(list)
    anns_by_image = defaultdict(list)
    all_image_ids = [int(image["id"]) for image in coco["images"]]
    for ann in coco["annotations"]:
        anns_by_class[int(ann["category_id"])].append(ann)
        anns_by_image[int(ann["image_id"])].append(ann)

    images_by_clean_class = defaultdict(list)
    for image_id, anns in anns_by_image.items():
        if len(anns) == 1:
            images_by_clean_class[int(anns[0]["category_id"])].append(image_id)

    selected = []
    selected_set = set()
    for cid in sorted(anns_by_class):
        # Prefer images that contain exactly one annotation from the target
        # class. This keeps the intended K-shot budget close to K instances per
        # class and avoids accidentally turning a split into a many-instance
        # setting when an image contains dense repeated defects.
        clean_candidates = list(images_by_clean_class[cid])
        rng.shuffle(clean_candidates)
        picked = 0
        for image_id in clean_candidates:
            if image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            picked += 1
            if picked >= shot:
                break
        if picked >= shot:
            continue

        fallback = sorted(
            {int(ann["image_id"]) for ann in anns_by_class[cid]},
            key=lambda image_id: (
                len(anns_by_image[image_id]),
                image_id,
            ),
        )
        rng.shuffle(fallback)
        fallback.sort(key=lambda image_id: len(anns_by_image[image_id]))
        for image_id in fallback:
            if image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            picked += 1
            if picked >= shot:
                break
        if picked < shot:
            raise RuntimeError(
                f"Only picked {picked} images for class {cid}, shot={shot}")

    bg_candidates = [
        image_id for image_id in all_image_ids
        if image_id not in selected_set and not anns_by_image.get(image_id)
    ]
    rng.shuffle(bg_candidates)
    selected.extend(bg_candidates[:background_images])
    selected_set.update(bg_candidates[:background_images])

    # Keep output order deterministic and close to the original COCO order.
    train_ids = [image_id for image_id in all_image_ids if image_id in selected_set]
    val_ids = [image_id for image_id in all_image_ids if image_id not in selected_set]
    return train_ids, val_ids


def convert_one(jsonl_path: Path, output_path: Path,
                categories: list[dict]) -> tuple[int, int]:
    images = []
    annotations = []
    ann_id = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for img_id, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            images.append({
                "id": img_id,
                "width": int(record["width"]),
                "height": int(record["height"]),
                "file_name": record["filename"],
            })

            for inst in record.get("detection", {}).get("instances", []):
                x1, y1, x2, y2 = [float(v) for v in inst["bbox"]]
                w = max(x2 - x1, 0.0)
                h = max(y2 - y1, 0.0)
                if w <= 0 or h <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(inst["label"]),
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

    coco_out = {
        "info": {
            "description": "Few-shot DSA defect train split converted from ODVG JSONL",
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False, indent=2)

    return len(images), len(annotations)



def check_categories(coco: dict) -> None:
    expected = dict(enumerate(CLASSES))
    actual = {int(cat['id']): cat['name'] for cat in coco['categories']}
    if actual != expected:
        raise ValueError(f'Expected category IDs and names {expected}; found {actual}. '
                         'This preparation recipe is specific to the DSA class schema.')


def check_coco(coco: dict) -> None:
    check_categories(coco)
    image_ids = [int(image['id']) for image in coco['images']]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError('COCO contains duplicate image IDs')
    ids = set(image_ids)
    for ann in coco['annotations']:
        if int(ann['image_id']) not in ids or int(ann['category_id']) not in range(5):
            raise ValueError(f'Invalid COCO image/category reference in annotation {ann["id"]}')
        if len(ann['bbox']) != 4 or ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
            raise ValueError(f'Invalid COCO bbox in annotation {ann["id"]}')
    for image in coco['images']:
        if image['width'] <= 0 or image['height'] <= 0:
            raise ValueError(f'Invalid image size for {image["file_name"]}')


def check_label_map(root: Path, create: bool = False) -> None:
    path = root / 'label_map.json'
    expected = {str(index): name for index, name in enumerate(CLASSES)}
    if not path.exists() and create:
        write_json(path, expected, overwrite=False)
    if load_coco(path) != expected:
        raise ValueError('label_map.json must map string IDs "0" through "4" '
                         'to the five documented class names')


def verify_split(root: Path, shot: int, seed: int, check_images: bool = False) -> None:
    check_label_map(root)
    split_dir = root / 'few_shot_dataset'
    train_path = split_dir / f'train_{shot}shot_seed{seed}.jsonl'
    val_path = split_dir / f'val_{shot}shot_seed{seed}.json'
    records = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f'Empty training file: {train_path}')
    val = load_coco(val_path)
    check_coco(val)
    train_names = [record['filename'] for record in records]
    val_names = [image['file_name'] for image in val['images']]
    if len(train_names) != len(set(train_names)) or len(val_names) != len(set(val_names)):
        raise ValueError('Duplicate filenames found within a split')
    overlap = set(train_names) & set(val_names)
    if overlap:
        raise ValueError(f'Training and validation overlap: {sorted(overlap)[:5]}')
    counts = Counter()
    for record in records:
        if int(record['width']) <= 0 or int(record['height']) <= 0:
            raise ValueError(f'Invalid image size for {record["filename"]}')
        for instance in record['detection']['instances']:
            label = int(instance['label'])
            if label not in range(5):
                raise ValueError(f'Invalid OD label {label} in {record["filename"]}')
            counts[label] += 1
            x1, y1, x2, y2 = instance['bbox']
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid OD bbox in {record["filename"]}')
        grounding = record['grounding']
        caption = grounding['caption']
        for region in grounding['regions']:
            if not region['phrase']:
                raise ValueError(f'Empty grounding phrase in {record["filename"]}')
            for start, end in region['tokens_positive']:
                if not 0 <= start < end <= len(caption):
                    raise ValueError(f'Invalid grounding token span in {record["filename"]}')
    if check_images:
        for name in train_names + val_names:
            if not (root / 'images' / name).is_file():
                raise FileNotFoundError(root / 'images' / name)
    support_counts = {CLASSES[index]: counts[index] for index in range(5)}
    print(f'{shot}-shot seed {seed}: train={len(records)}, val={len(val_names)}, '
          f'support instances={support_counts}; no filename overlap')


def prepare_splits(args: argparse.Namespace, root: Path) -> None:
    # The historical partitions are inputs to reproduction, not outputs of this
    # later extension script. Refuse to overwrite or relabel seeds 1--3.
    if any(seed <= 3 for seed in args.seeds):
        raise ValueError('Seeds 1--3 are historical partitions: obtain their existing files. '
                         'This generator supports seeds >= 4 and does not reconstruct them.')
    coco = load_coco(root / 'custom_coco_dataset.json')
    check_coco(coco)
    check_label_map(root, create=True)
    split_dir = root / 'few_shot_dataset'
    for shot in args.shots:
        for seed in args.seeds:
            paths = [split_dir / f'train_{shot}shot_seed{seed}.jsonl',
                     split_dir / f'train_{shot}shot_seed{seed}_coco.json',
                     split_dir / f'val_{shot}shot_seed{seed}.json']
            if any(path.exists() for path in paths) and not args.overwrite:
                if not all(path.exists() for path in paths):
                    raise FileExistsError(f'Partial split exists for shot={shot}, seed={seed}; '
                                          'keep the original files or explicitly use --overwrite.')
                print(f'Skip existing {shot}-shot seed {seed} (no files rewritten)')
                continue
            train_ids, val_ids = create_split(coco, shot, seed, args.background_images)
            train_coco = reindex_coco(coco, train_ids)
            val_coco = reindex_coco(coco, val_ids)
            write_jsonl(paths[0], coco_to_jsonl_records(train_coco), args.overwrite)
            write_json(paths[1], train_coco, args.overwrite)
            write_json(paths[2], val_coco, args.overwrite)
            verify_split(root, shot, seed)


def convert_splits(args: argparse.Namespace, root: Path) -> None:
    coco = load_coco(root / 'custom_coco_dataset.json')
    check_categories(coco)
    for shot in args.shots:
        for seed in args.seeds:
            source = root / 'few_shot_dataset' / f'train_{shot}shot_seed{seed}.jsonl'
            destination = source.with_name(source.stem + '_coco.json')
            if destination.exists() and not args.overwrite:
                print(f'Skip existing {destination.name}')
                continue
            images, annotations = convert_one(source, destination, coco['categories'])
            print(f'{source.name} -> {destination.name}: {images} images, {annotations} annotations')


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action, help_text in (
            ('verify', 'check existing support/validation annotations and optional image paths'),
            ('splits', 'generate additional splits (seeds >= 4) from full COCO annotations'),
            ('convert', 'convert existing support ODVG JSONL to COCO without resampling')):
        sub = actions.add_parser(action, help=help_text)
        sub.add_argument('--data-root', default=os.environ.get('DSA_DATA_ROOT', str(REPO_ROOT / 'data/dsa')))
        sub.add_argument('--shots', type=positive_int, nargs='+', default=[1, 3, 5, 10])
        sub.add_argument('--seeds', type=positive_int, nargs='+',
                         default=list(range(4, 11)) if action == 'splits' else [1, 2, 3])
        if action == 'verify':
            sub.add_argument('--check-images', action='store_true', help='also require each image file to exist')
        else:
            sub.add_argument('--overwrite', action='store_true', help='explicitly replace generated annotations')
        if action == 'splits':
            sub.add_argument('--background-images', type=int, default=2,
                             help='background support images; original recipe uses 2')
    args = parser.parse_args()
    if getattr(args, 'background_images', 0) < 0:
        parser.error('--background-images cannot be negative')
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if args.action == 'verify':
        for shot in args.shots:
            for seed in args.seeds:
                verify_split(root, shot, seed, args.check_images)
    elif args.action == 'splits':
        prepare_splits(args, root)
    else:
        convert_splits(args, root)


if __name__ == '__main__':
    main()
