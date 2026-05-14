"""
Select better support images for few-shot fine-tuning.

Strategy:
1. Use the 5-shot support set as a baseline (known to work well)
2. Replace novel-class (ship) images with the best available from the training pool
3. For base classes, reuse the proven 5-shot images
4. Selection criteria: large bbox area, good aspect ratio, not truncated, diversity
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from copy import deepcopy


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def score_ship_annotation(ann, img, img_size_weight=1.0, area_weight=2.0,
                           truncation_penalty=10.0, extreme_aspect_penalty=5.0):
    """
    Score a ship annotation for suitability as a support image.
    Higher score = better support image.

    Criteria:
    - Larger bbox area is better (more visible features)
    - Larger image resolution is better
    - Not truncated at image edges
    - Reasonable aspect ratio (not extremely thin)
    """
    x, y, w, h = ann['bbox']
    area = ann['area']
    img_w, img_h = img['width'], img['height']

    # Bbox area score (log scale to avoid huge ships dominating)
    area_score = np.log(area + 1) / np.log(10)  # normalize

    # Image size score (prefer larger images)
    img_size_score = np.log(min(img_w, img_h) + 1) / np.log(10)

    # Truncation check: bbox should not be at image edge
    truncation = 0
    # Check if bbox touches image edges (with margin)
    margin = 5
    if x < margin:
        truncation += (margin - x) / margin
    if y < margin:
        truncation += (margin - y) / margin
    if x + w > img_w - margin:
        truncation += (x + w - img_w + margin) / margin
    if y + h > img_h - margin:
        truncation += (y + h - img_h + margin) / margin

    # Aspect ratio: prefer objects that aren't extremely thin
    aspect = max(w, h) / max(min(w, h), 1)
    aspect_score = 0
    if aspect > 5:
        aspect_score = (aspect - 5) * 0.5  # penalty for very thin objects

    # Total score
    score = (area_weight * area_score +
             img_size_weight * img_size_score -
             truncation_penalty * truncation -
             extreme_aspect_penalty * aspect_score)

    return score


def select_best_ships(base_train, n_shots, diversity_weight=0.15):
    """
    Select the best n_shots ship images from the training pool.

    Uses a greedy approach: first pick the highest-scoring image,
    then pick subsequent images that balance quality and diversity.

    Filters applied before selection:
    - Min bbox area: 2000
    - Min bbox width and height: 30 pixels
    - Aspect ratio < 8 (not extremely thin)
    - Bbox not severely truncated at image edges

    Args:
        base_train: loaded base_train.json
        n_shots: number of ship images to select
        diversity_weight: how much to weight diversity vs quality

    Returns:
        list of (annotation, image) tuples
    """
    MIN_AREA = 2000
    MIN_SIDE = 30
    MAX_ASPECT = 8.0
    MAX_TRUNCATION = 0.3

    # Build lookup
    img_by_id = {img['id']: img for img in base_train['images']}

    # Get all ship annotations, filtered by quality criteria
    ship_entries = []
    for ann in base_train['annotations']:
        if ann['category_id'] != 0:  # ship
            continue

        x, y, w, h = ann['bbox']
        area = ann['area']

        # Quality filters
        if area < MIN_AREA:
            continue
        if w < MIN_SIDE or h < MIN_SIDE:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > MAX_ASPECT:
            continue

        # Truncation filter
        img = img_by_id[ann['image_id']]
        img_w, img_h = img['width'], img['height']
        margin = 3
        trunc = 0
        if x < margin:
            trunc += 1
        if y < margin:
            trunc += 1
        if x + w > img_w - margin:
            trunc += 1
        if y + h > img_h - margin:
            trunc += 1
        if trunc >= 2:  # allow at most 1 edge touch
            continue

        score = score_ship_annotation(ann, img)
        ship_entries.append((score, ann, img))

    print(f"  (Filtered to {len(ship_entries)} quality ship candidates "
          f"from {len(base_train['annotations'])} total annotations)")

    # Sort by score
    ship_entries.sort(key=lambda x: x[0], reverse=True)

    selected = []
    selected_features = []  # (aspect, log_area, log_img_size) for diversity

    if n_shots == 1:
        best = ship_entries[0]
        selected.append((best[1], best[2]))
    else:
        # First pick the best
        best = ship_entries[0]
        selected.append((best[1], best[2]))
        sel_ann, sel_img = best[1], best[2]
        sel_aspect = max(sel_ann['bbox'][2], sel_ann['bbox'][3]) / max(min(sel_ann['bbox'][2], sel_ann['bbox'][3]), 1)
        selected_features.append({
            'aspect': sel_aspect,
            'log_area': np.log(sel_ann['area']),
            'log_size': np.log(sel_img['width'] * sel_img['height']),
            'relative_x': sel_ann['bbox'][0] / sel_img['width'],
            'relative_y': sel_ann['bbox'][1] / sel_img['height'],
        })

        for _ in range(n_shots - 1):
            best_candidate = None
            best_combined_score = -float('inf')

            for score, ann, img in ship_entries:
                # Skip already selected (by image_id)
                if img['id'] in [s[1]['id'] for s in selected]:
                    continue

                # Feature vector for this candidate
                this_aspect = max(ann['bbox'][2], ann['bbox'][3]) / max(min(ann['bbox'][2], ann['bbox'][3]), 1)
                this_feat = {
                    'aspect': this_aspect,
                    'log_area': np.log(ann['area']),
                    'log_size': np.log(img['width'] * img['height']),
                    'relative_x': ann['bbox'][0] / img['width'],
                    'relative_y': ann['bbox'][1] / img['height'],
                }

                # Compute diversity against all selected
                diversity_scores = []
                for sf in selected_features:
                    aspect_div = abs(np.log(sf['aspect'] + 0.1) - np.log(this_feat['aspect'] + 0.1))
                    area_div = abs(sf['log_area'] - this_feat['log_area'])
                    size_div = abs(sf['log_size'] - this_feat['log_size'])
                    pos_div = np.sqrt((sf['relative_x'] - this_feat['relative_x'])**2 +
                                       (sf['relative_y'] - this_feat['relative_y'])**2)
                    pair_div = aspect_div * 2.0 + area_div * 0.8 + size_div * 0.5 + pos_div * 1.5
                    diversity_scores.append(pair_div)

                avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
                # Normalize: diversity ranges from 0 (identical) to 8+ (very different)
                # Scale so that diversity adds at most ~30% of the base score
                combined_score = score * (1.0 + diversity_weight * min(avg_diversity, 4.0))

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_candidate = (ann, img)

            if best_candidate:
                ann, img = best_candidate
                selected.append((ann, img))
                this_aspect = max(ann['bbox'][2], ann['bbox'][3]) / max(min(ann['bbox'][2], ann['bbox'][3]), 1)
                selected_features.append({
                    'aspect': this_aspect,
                    'log_area': np.log(ann['area']),
                    'log_size': np.log(img['width'] * img['height']),
                    'relative_x': ann['bbox'][0] / img['width'],
                    'relative_y': ann['bbox'][1] / img['height'],
                })

    return selected


def get_category_annotations_from_5shot(s5_data, category_id, n_shots):
    """
    Extract up to n_shots annotations (and their images) for a given category
    from the 5-shot support set.

    Returns list of (annotation, image) tuples.
    """
    anns = [a for a in s5_data['annotations'] if a['category_id'] == category_id]
    img_by_id = {img['id']: img for img in s5_data['images']}

    # Group annotations by image_id to handle multiple anns per image
    img_groups = {}
    for a in anns:
        img_id = a['image_id']
        if img_id not in img_groups:
            img_groups[img_id] = []
        img_groups[img_id].append((a, img_by_id[img_id]))

    # Take first n_shots annotations
    selected = []
    for img_id, pairs in img_groups.items():
        for pair in pairs:
            if len(selected) < n_shots:
                selected.append(pair)

    return selected[:n_shots]


def build_fewshot_json(ship_entries, base_entries_by_cat, s5_data, n_shots,
                        output_path):
    """
    Build a new FewShot JSON file combining selected ship images with
    base class images from the 5-shot set.

    Args:
        ship_entries: list of (annotation, image) for ship
        base_entries_by_cat: dict of cat_id -> list of (annotation, image)
        s5_data: the 5-shot data (for categories)
        n_shots: k-shot setting
        output_path: where to save
    """
    categories = deepcopy(s5_data['categories'])

    # Deduplicate images (same image could have multiple annotations)
    all_images = {}  # image_id -> image dict
    all_annotations = []
    next_ann_id = 1

    # Add ship entries
    for ann, img in ship_entries:
        img_id = img['id']
        if img_id not in all_images:
            all_images[img_id] = {
                'file_name': img['file_name'],
                'height': img['height'],
                'width': img['width'],
                'id': img_id
            }
        new_ann = deepcopy(ann)
        new_ann['id'] = 1000000 + next_ann_id
        next_ann_id += 1
        all_annotations.append(new_ann)

    # Add base class entries
    for cat_id in range(1, 6):  # base classes: 1-5
        entries = base_entries_by_cat.get(cat_id, [])
        for ann, img in entries:
            img_id = img['id']
            if img_id not in all_images:
                all_images[img_id] = {
                    'file_name': img['file_name'],
                    'height': img['height'],
                    'width': img['width'],
                    'id': img_id
                }
            new_ann = deepcopy(ann)
            new_ann['id'] = 1000000 + next_ann_id
            next_ann_id += 1
            all_annotations.append(new_ann)

    output = {
        'images': list(all_images.values()),
        'annotations': all_annotations,
        'categories': categories
    }

    save_json(output, output_path)

    # Print summary
    print(f"\nGenerated {output_path}:")
    print(f"  {len(output['images'])} images, {len(output['annotations'])} annotations")
    for cat in categories:
        count = sum(1 for a in all_annotations if a['category_id'] == cat['id'])
        print(f"  {cat['name']}: {count} annotations")


def main():
    parser = argparse.ArgumentParser(
        description='Select better support images for few-shot detection')
    parser.add_argument('--split', type=str, default='split2',
                        help='Dataset split (split1 or split2)')
    parser.add_argument('--data-root', type=str, default='data/sardet100k',
                        help='Root data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for new JSON files')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3],
                        help='Shot settings to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = args.data_root
    split = args.split

    if args.output_dir is None:
        output_dir = os.path.join(data_root, split)
    else:
        output_dir = args.output_dir

    # Load data
    base_train_path = os.path.join(data_root, split, 'base_train.json')
    s5_path = os.path.join(data_root, split, 'FewShot_5shot_train_seed0.json')

    print(f"Loading base_train from: {base_train_path}")
    base_train = load_json(base_train_path)

    print(f"Loading 5-shot from: {s5_path}")
    s5_data = load_json(s5_path)

    # Build image lookup for base_train
    bt_img_by_id = {img['id']: img for img in base_train['images']}

    # For each shot setting, select ships and build JSON
    for n_shots in args.shots:
        print(f"\n{'='*60}")
        print(f"Processing {n_shots}-shot...")
        print(f"{'='*60}")

        # Select best ship images
        ship_entries = select_best_ships(base_train, n_shots)
        print(f"\nSelected ship images for {n_shots}-shot:")
        for i, (ann, img) in enumerate(ship_entries):
            bbox = ann['bbox']
            aspect = bbox[2] / max(bbox[3], 1)
            print(f"  [{i+1}] {img['file_name']} ({img['width']}x{img['height']}), "
                  f"area={ann['area']:.0f}, bbox={bbox}, aspect={aspect:.2f}")

        # Get base class entries from 5-shot
        base_entries = {}
        for cat_id in range(1, 6):
            entries = get_category_annotations_from_5shot(s5_data, cat_id, n_shots)
            base_entries[cat_id] = entries

        # Build output JSON
        output_path = os.path.join(
            output_dir, f'FewShot_{n_shots}shot_train_seed{args.seed}.json')
        build_fewshot_json(ship_entries, base_entries, s5_data, n_shots,
                           output_path)


if __name__ == '__main__':
    main()
