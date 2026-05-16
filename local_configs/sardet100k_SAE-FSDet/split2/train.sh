#!/bin/bash
# sardet100k_SAE-FSDet split2 training script

# Step 1: Base Training
bash tools/detection/dist_train_v2.sh local_configs/sardet100k_SAE-FSDet/split2/SAE-FSDet_sardet100k_split2_4xb2_BT.py 4

# Step 2: Initialize bbox head and bbox regressor for all classes
python tools/detection/misc/initialize_bbox_head.py \
    --src1 work_dirs/SAE-FSDet/sardet100k/split2/4xb2_BT/latest.pth \
    --method random_init \
    --param-name roi_head.bbox_head.fc_cls roi_head.bbox_head.fc_reg \
    --save-dir work_dirs/SAE-FSDet/sardet100k/split2/4xb2_BT/ \
    --sardet100k --sardet100k_split 2

# Step 3: Fine-tuning
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sardet100k_SAE-FSDet/split2/SAE-FSDet_sardet100k_split2_4xb2_${shot}shot_FT.py"
    echo ">>> Running Task: sardet100k_SAE-FSDet/split2 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 4
done
