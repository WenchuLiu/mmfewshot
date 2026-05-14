"""
Select better support images for few-shot fine-tuning.

Strategy:
1. For novel class (ship): select best images from entire training pool
2. For base classes: select best images from training pool (not just 5-shot)
3. Selection: large bbox, good aspect ratio, not truncated, diverse coverage
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
    """Score an annotation for suitability as a support image."""
    x, y, w, h = ann['bbox']
    area = ann['area']
    img_w, img_h = img['width'], img['height']

    area_score = np.log(area + 1) / np.log(10)
    img_size_score = np.log(min(img_w, img_h) + 1) / np.log(10)

    margin = 5
    truncation = 0
    if x < margin:
        truncation += (margin - x) / margin
    if y < margin:
        truncation += (margin - y) / margin
    if x + w > img_w - margin:
        truncation += (x + w - img_w + margin) / margin
    if y + h > img_h - margin:
        truncation += (y + h - img_h + margin) / margin

    aspect = max(w, h) / max(min(w, h), 1)
    aspect_penalty = 0
    if aspect > 5:
        aspect_penalty = (aspect - 5) * 0.5

    return (2.0 * area_score + 1.0 * img_size_score -
            10.0 * truncation - 5.0 * aspect_penalty)


def filter_quality_annotations(base_train, category_id):
    """Filter annotations to only quality ones."""
    MIN_AREA = 2000
    MIN_SIDE = 30
    MAX_ASPECT = 8.0

    img_by_id = {img['id']: img for img in base_train['images']}
    results = []

    for ann in base_train['annotations']:
        if ann['category_id'] != category_id:
            continue
        x, y, w, h = ann['bbox']
        if ann['area'] < MIN_AREA or w < MIN_SIDE or h < MIN_SIDE:
            continue
        if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
            continue

        img = img_by_id[ann['image_id']]
        img_w, img_h = img['width'], img['height']
        margin = 3
        trunc = sum([x < margin, y < margin,
                      x + w > img_w - margin, y + h > img_h - margin])
        if trunc >= 2:
            continue

        score = score_annotation(ann, img)
        results.append((score, ann, img))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def select_diverse(candidates, n_shots, diversity_weight=0.15):
    """Select n_shots diverse entries from sorted candidates."""
    if n_shots >= len(candidates):
        return [(a, i) for _, a, i in candidates]

    selected = []
    selected_feats = []

    best = candidates[0]
    selected.append((best[1], best[2]))
    _add_features(selected_feats, best[1], best[2])

    for _ in range(n_shots - 1):
        best_candidate = None
        best_combined = -float('inf')

        for score, ann, img in candidates:
            if img['id'] in [s[1]['id'] for s in selected]:
                continue

            this_feat = _get_features(ann, img)
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
            _add_features(selected_feats, best_candidate[0], best_candidate[1])

    return selected


def _get_features(ann, img):
    w, h = ann['bbox'][2], ann['bbox'][3]
    return {
        'aspect': max(w, h) / max(min(w, h), 1),
        'log_area': np.log(ann['area']),
        'log_size': np.log(img['width'] * img['height']),
        'rx': ann['bbox'][0] / img['width'],
        'ry': ann['bbox'][1] / img['height'],
    }


def _add_features(feat_list, ann, img):
    feat_list.append(_get_features(ann, img))


def build_fewshot_json(all_selected, categories_data, output_path):
    """Build FewShot JSON from selected entries per category."""
    all_images = {}
    all_annotations = []
    next_ann_id = 1

    for cat_id, entries in all_selected.items():
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
        'categories': deepcopy(categories_data)
    }
    save_json(output, output_path)

    print(f"  -> {output_path}: {len(output['images'])} images, "
          f"{len(output['annotations'])} annotations")
    for cat in categories_data:
        cnt = sum(1 for a in all_annotations if a['category_id'] == cat['id'])
        print(f"     {cat['name']}: {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description='Select better support images')
    parser.add_argument('--split', default='split2')
    parser.add_argument('--data-root', default='data/sardet100k')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--improve-base', action='store_true',
                        help='Also reselect base class images (default: from 5-shot)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_root = args.data_root
    split = args.split
    output_dir = args.output_dir or os.path.join(data_root, split)

    base_train = load_json(os.path.join(data_root, split, 'base_train.json'))
    s5_data = load_json(os.path.join(data_root, split,
                                      'FewShot_5shot_train_seed0.json'))
    categories = s5_data['categories']
    cat_names = {c['id']: c['name'] for c in categories}

    # Build image lookup for 5-shot
    s5_img_by_id = {img['id']: img for img in s5_data['images']}

    for n_shots in args.shots:
        print(f"\n{'='*60}")
        print(f"Generating {n_shots}-shot support set (seed={args.seed})")
        print(f"{'='*60}")

        all_selected = {}

        for cat in categories:
            cat_id = cat['id']
            cat_name = cat['name']

            if args.improve_base or cat_id == 0:  # ship (novel) always improved
                candidates = filter_quality_annotations(base_train, cat_id)
                print(f"\n  {cat_name}: {len(candidates)} quality candidates")
                selected = select_diverse(candidates, n_shots)
            else:
                # Reuse 5-shot base class images
                anns = [a for a in s5_data['annotations']
                        if a['category_id'] == cat_id]
                selected = []
                for a in anns[:n_shots]:
                    img = s5_img_by_id[a['image_id']]
                    if (a, img) not in selected:
                        selected.append((a, img))
                selected = selected[:n_shots]

            # Print selections
            for i, (ann, img) in enumerate(selected):
                bbox = ann['bbox']
                aspect = bbox[2] / max(bbox[3], 1)
                print(f"    [{i+1}] {img['file_name']} "
                      f"({img['width']}x{img['height']}), "
                      f"area={ann['area']:.0f}, aspect={aspect:.2f}")

            all_selected[cat_id] = selected

        output_path = os.path.join(
            output_dir, f'FewShot_{n_shots}shot_train_seed{args.seed}.json')
        build_fewshot_json(all_selected, categories, output_path)


if __name__ == '__main__':
    main()
