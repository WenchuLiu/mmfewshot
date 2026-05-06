import argparse
import json
import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm  # 如果没有安装，可以使用 pip install tqdm，或者删除相关代码
'''
python tools/detection/vis_json.py \
    data/few_shot_ann/sardet100k/sar_3/FewShot_5shot_train_seed0.json \
    /data/SOI_Det/SARDet_100K/JPEGImages/train/ \
    --output-dir ./vis_results_5shot \
    -c aircraft
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO/MMRotate JSON Dataset')
    
    # --- 基础路径 ---
    parser.add_argument('json_path', help='Path to the annotation JSON file')
    parser.add_argument('img_dir', help='Path to the image directory')
    
    # --- 筛选与控制 ---
    parser.add_argument('--category', '-c', type=str, default=None, 
                        help='Filter images: Only visualize images containing this specific class name (e.g., "plane")')
    parser.add_argument('--output-dir', '-o', default=None, 
                        help='Directory to save visualized images. If None, show window.')
    parser.add_argument('--vis-type', '-t', choices=['bbox', 'poly', 'both'], default='poly', 
                        help='Type of visualization: "bbox" (HBB), "poly" (OBB/Segmentation), or "both"')
    parser.add_argument('--num-imgs', '-n', type=int, default=None, help='Number of images to visualize')
    parser.add_argument('--shuffle', action='store_true', help='Randomly shuffle images')
    parser.add_argument('--show-label', action='store_true', default=True, help='Show text labels')
    parser.add_argument('--score-thr', type=float, default=0.0, help='Score threshold (if json has scores)')

    args = parser.parse_args()
    return args

def get_random_color(seed=None):
    if seed:
        random.seed(seed)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def main():
    args = parse_args()

    # 1. 检查文件
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")
    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"Image dir not found: {args.img_dir}")

    # 2. 加载 JSON
    print(f"Loading JSON from {args.json_path} ...")
    with open(args.json_path, 'r') as f:
        coco = json.load(f)
    
    # 建立索引: ID <-> Name
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    cat_name_to_id = {cat['name']: cat['id'] for cat in coco['categories']} # 反向索引用于查询

    # 建立索引: ImageID -> Annotations
    img_id_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # 3. 核心修改：按类别过滤图片列表
    images = coco['images']
    
    if args.category:
        target_name = args.category
        if target_name not in cat_name_to_id:
            print(f"\n[Error] Class '{target_name}' not found in dataset!")
            print(f"Available classes: {list(cat_name_to_id.keys())}")
            return
        
        target_cat_id = cat_name_to_id[target_name]
        print(f"Filtering images containing class: '{target_name}' (ID: {target_cat_id})...")
        
        # 找出所有包含该类别的 image_id
        valid_img_ids = set()
        for ann in coco['annotations']:
            if ann['category_id'] == target_cat_id:
                valid_img_ids.add(ann['image_id'])
        
        # 过滤 images 列表
        original_count = len(images)
        images = [img for img in images if img['id'] in valid_img_ids]
        print(f"Found {len(images)} images (out of {original_count}).")

        if len(images) == 0:
            print("No images found with this class. Exiting.")
            return

    # 4. 后处理：打乱和截断
    if args.shuffle:
        random.shuffle(images)
    
    if args.num_imgs is not None:
        images = images[:args.num_imgs]

    # 准备颜色
    cat_colors = {cat_id: get_random_color(cat_id) for cat_id in cat_id_to_name}

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving to {args.output_dir} ...")
    else:
        print("Press 'Space' for next, 'q' to quit.")

    # 5. 循环可视化
    for i, img_info in enumerate(tqdm(images, desc="Visualizing")):
        img_name = img_info['file_name']
        img_id = img_info['id']
        img_path = os.path.join(args.img_dir, img_name)

        if not os.path.exists(img_path):
            # 兼容有些数据集带文件夹前缀的情况
            img_path_alt = os.path.join(args.img_dir, os.path.basename(img_name))
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                continue

        img = cv2.imread(img_path)
        if img is None: continue

        anns = img_id_to_anns.get(img_id, [])

        # 绘制标注
        for ann in anns:
            if 'score' in ann and ann['score'] < args.score_thr:
                continue

            cat_id = ann['category_id']
            # 可选：如果你只想看目标类的框，把下面这两行取消注释
            # if args.category and cat_id != target_cat_id:
            #     continue 

            color = cat_colors.get(cat_id, (0, 255, 0))
            label_name = cat_id_to_name.get(cat_id, 'unknown')
            
            has_seg = 'segmentation' in ann and ann['segmentation']
            
            # 画多边形
            if (args.vis_type in ['poly', 'both']) and has_seg:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)
            
            # 画水平框
            if args.vis_type in ['bbox', 'both'] or (args.vis_type == 'poly' and not has_seg):
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # 画文字
            if args.show_label:
                # 确定文字位置
                if has_seg and args.vis_type != 'bbox':
                    text_pos = (int(ann['segmentation'][0][0]), int(ann['segmentation'][0][1]))
                else:
                    x, y, _, _ = map(int, ann['bbox'])
                    text_pos = (x, y - 5)
                
                label_text = f"{label_name}"
                if 'score' in ann:
                    label_text += f" {ann['score']:.2f}"
                
                # 文字描边，防止看不清
                cv2.putText(img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 保存或显示
        if args.output_dir:
            save_path = os.path.join(args.output_dir, os.path.basename(img_name))
            cv2.imwrite(save_path, img)
        else:
            h, w = img.shape[:2]
            scale = 1000 / max(h, w)
            if scale < 1:
                img_show = cv2.resize(img, None, fx=scale, fy=scale)
            else:
                img_show = img
            
            cv2.imshow('Visualizer', img_show)
            key = cv2.waitKey(0)
            if key == ord('q'): break

    if not args.output_dir:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()