"""
Select better support images for few-shot fine-tuning on sardet100k split2.

Key insight: split2's novel class is 'ship', but ship images are in split1's
base_train.json (since ship is a base class for split1). The training images
are shared across splits.

Category mapping:
  split1 base_train: {0: ship, 1: car, 2: tank, 3: bridge, 4: harbor}
  split2 base_train: {0: aircraft, 1: car, 2: tank, 3: bridge, 4: harbor}
  FewShot JSONs:     {0: ship, 1: aircraft, 2: car, 3: tank, 4: bridge, 5: harbor}

Strategy:
- Ship (novel): select from split1 base_train (61287 ship images)
- Base classes: select from split2 base_train (or reuse 5-shot images)
"""

import argparse
import json
import os
import numpy as np
from copy import deepcopy


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def score_annotation(ann, img):
    x, y, w, h = ann['bbox']
    area = ann['area']
    img_w, img_h = img['width'], img['height']

    area_score = np.log(area + 1) / np.log(10)
    img_size_score = np.log(min(img_w, img_h) + 1) / np.log(10)

    margin = 5
    trunc = 0
    if x < margin:
        trunc += (margin - x) / margin
    if y < margin:
        trunc += (margin - y) / margin
    if x + w > img_w - margin:
        trunc += (x + w - img_w + margin) / margin
    if y + h > img_h - margin:
        trunc += (y + h - img_h + margin) / margin

    aspect = max(w, h) / max(min(w, h), 1)
    aspect_penalty = 0
    if aspect > 5:
        aspect_penalty = (aspect - 5) * 0.5

    return (2.0 * area_score + 1.0 * img_size_score -
            10.0 * trunc - 5.0 * aspect_penalty)


def filter_quality(candidates_data, category_id, test_files=None):
    """Extract quality candidates for a category from a dataset."""
    MIN_AREA = 2000
    MIN_SIDE = 30
    MAX_ASPECT = 8.0

    img_by_id = {img['id']: img for img in candidates_data['images']}
    results = []

    for ann in candidates_data['annotations']:
        if ann['category_id'] != category_id:
            continue

        x, y, w, h = ann['bbox']
        if ann['area'] < MIN_AREA or w < MIN_SIDE or h < MIN_SIDE:
            continue
        if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
            continue

        img = img_by_id[ann['image_id']]

        # Exclude images in test set
        if test_files and img['file_name'] in test_files:
            continue

        img_w, img_h = img['width'], img['height']
        margin = 3
        trunc_cnt = sum([x < margin, y < margin,
                         x + w > img_w - margin, y + h > img_h - margin])
        if trunc_cnt >= 2:
            continue

        score = score_annotation(ann, img)
        results.append((score, ann, img))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def get_features(ann, img):
    w, h = ann['bbox'][2], ann['bbox'][3]
    return {
        'aspect': max(w, h) / max(min(w, h), 1),
        'log_area': np.log(ann['area']),
        'log_size': np.log(img['width'] * img['height']),
        'rx': ann['bbox'][0] / img['width'],
        'ry': ann['bbox'][1] / img['height'],
    }


def select_diverse(candidates, n_shots, diversity_weight=0.15):
    if n_shots >= len(candidates):
        return [(a, i) for _, a, i in candidates]

    selected = []
    selected_feats = []

    best = candidates[0]
    selected.append((best[1], best[2]))
    selected_feats.append(get_features(best[1], best[2]))

    for _ in range(n_shots - 1):
        best_candidate = None
        best_combined = -float('inf')

        for score, ann, img in candidates:
            if img['id'] in [s[1]['id'] for s in selected]:
                continue

            this_feat = get_features(ann, img)
            div_scores = []
            for sf in selected_feats:
                aspect_div = abs(np.log(sf['aspect'] + 0.1) -
                                 np.log(this_feat['aspect'] + 0.1))
                area_div = abs(sf['log_area'] - this_feat['log_area'])
                size_div = abs(sf['log_size'] - this_feat['log_size'])
                pos_div = np.sqrt((sf['rx'] - this_feat['rx'])**2 +
                                  (sf['ry'] - this_feat['ry'])**2)
                div_scores.append(aspect_div * 2.0 + area_div * 0.8 +
                                  size_div * 0.5 + pos_div * 1.5)

            avg_div = np.mean(div_scores)
            combined = score * (1.0 + diversity_weight * min(avg_div, 4.0))

            if combined > best_combined:
                best_combined = combined
                best_candidate = (ann, img)

        if best_candidate:
            selected.append(best_candidate)
            selected_feats.append(get_features(best_candidate[0], best_candidate[1]))

    return selected


def build_fewshot_json(all_selected, fewshot_categories, output_path):
    """Build output JSON with correct category IDs for FewShot format.
    fewshot_categories: list of {id, name} where id 0=ship, 1=aircraft, ...
    """
    all_images = {}
    all_annotations = []
    next_ann_id = 1

    for fewshot_cat_id, entries in all_selected.items():
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
            new_ann['category_id'] = fewshot_cat_id  # Use FewShot category ID
            next_ann_id += 1
            all_annotations.append(new_ann)

    output = {
        'images': list(all_images.values()),
        'annotations': all_annotations,
        'categories': deepcopy(fewshot_categories)
    }
    save_json(output, output_path)

    print(f"  -> {output_path}: {len(output['images'])} images, "
          f"{len(output['annotations'])} annotations")
    for cat in fewshot_categories:
        cnt = sum(1 for a in all_annotations if a['category_id'] == cat['id'])
        print(f"     {cat['name']}: {cnt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/sardet100k')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--improve-base', action='store_true',
                        help='Also reselect base class images')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_root = args.data_root

    # FewShot category mapping (the target format)
    # {0: ship, 1: aircraft, 2: car, 3: tank, 4: bridge, 5: harbor}
    fewshot_categories = [
        {'id': 0, 'name': 'ship'},
        {'id': 1, 'name': 'aircraft'},
        {'id': 2, 'name': 'car'},
        {'id': 3, 'name': 'tank'},
        {'id': 4, 'name': 'bridge'},
        {'id': 5, 'name': 'harbor'},
    ]

    # ----------------------------------------------------------------
    # 1. Ship pool: from split1 base_train (ship is base class in split1)
    #    split1 base_train: {0: ship, 1: car, 2: tank, 3: bridge, 4: harbor}
    # ----------------------------------------------------------------
    s1_base_train = load_json(os.path.join(data_root, 'split1', 'base_train.json'))

    # Get split2 test files to exclude
    test_files = set()
    for test_f in ['base_test.json', 'FewShot_test.json']:
        test_data = load_json(os.path.join(data_root, 'split2', test_f))
        for img in test_data['images']:
            test_files.add(img['file_name'])

    # Ship candidates from split1 base_train (category_id=0 in split1 = ship)
    ship_candidates = filter_quality(s1_base_train, 0, test_files)
    print(f"Ship quality candidates: {len(ship_candidates)}")

    # ----------------------------------------------------------------
    # 2. Base class pools: from split2 base_train
    #    split2 base_train: {0: aircraft, 1: car, 2: tank, 3: bridge, 4: harbor}
    # ----------------------------------------------------------------
    s2_base_train = load_json(os.path.join(data_root, 'split2', 'base_train.json'))

    # Map: FewShot cat_id -> (split2_base_train cat_id, class_name)
    base_class_map = {
        1: (0, 'aircraft'),  # FewShot 1=aircraft, split2_bt 0=aircraft
        2: (1, 'car'),       # FewShot 2=car, split2_bt 1=car
        3: (2, 'tank'),      # FewShot 3=tank, split2_bt 2=tank
        4: (3, 'bridge'),    # FewShot 4=bridge, split2_bt 3=bridge
        5: (4, 'harbor'),    # FewShot 5=harbor, split2_bt 4=harbor
    }

    # Load 5-shot data as fallback for base classes
    s5_data = load_json(os.path.join(data_root, 'split2',
                                      'FewShot_5shot_train_seed0.json'))
    s5_img_by_id = {img['id']: img for img in s5_data['images']}

    output_dir = args.output_dir or os.path.join(data_root, 'split2')

    for n_shots in args.shots:
        print(f"\n{'='*60}")
        print(f"Generating {n_shots}-shot support set (seed={args.seed})")
        print(f"{'='*60}")

        all_selected = {}

        # --- Ship (novel class, fewshot_id=0) ---
        ship_selected = select_diverse(ship_candidates, n_shots)
        print(f"\n  ship (novel):")
        for i, (ann, img) in enumerate(ship_selected):
            bbox = ann['bbox']
            aspect = bbox[2] / max(bbox[3], 1)
            print(f"    [{i+1}] {img['file_name']} ({img['width']}x{img['height']}), "
                  f"area={ann['area']:.0f}, aspect={aspect:.2f}")
        all_selected[0] = ship_selected

        # --- Base classes ---
        for fewshot_cat_id, (s2bt_cat_id, cat_name) in base_class_map.items():
            if args.improve_base:
                candidates = filter_quality(s2_base_train, s2bt_cat_id)
                selected = select_diverse(candidates, n_shots)
            else:
                # Reuse from 5-shot
                anns = [a for a in s5_data['annotations']
                        if a['category_id'] == fewshot_cat_id]
                selected = []
                for a in anns:
                    img = s5_img_by_id[a['image_id']]
                    if (a, img) not in selected:
                        selected.append((a, img))
                selected = selected[:n_shots]

            print(f"\n  {cat_name} (base): {len(selected)} selected")
            for i, (ann, img) in enumerate(selected):
                bbox = ann['bbox']
                aspect = bbox[2] / max(bbox[3], 1)
                print(f"    [{i+1}] {img['file_name']} ({img['width']}x{img['height']}), "
                      f"area={ann['area']:.0f}, aspect={aspect:.2f}")

            all_selected[fewshot_cat_id] = selected

        output_path = os.path.join(
            output_dir, f'FewShot_{n_shots}shot_train_seed{args.seed}.json')
        build_fewshot_json(all_selected, fewshot_categories, output_path)


if __name__ == '__main__':
    main()
