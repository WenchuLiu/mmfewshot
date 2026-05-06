#!/bin/bash
# sardet100k_GFSDet split2 training script

# Prerequisite: TFA Base Training (if not already done)
# bash tools/detection/dist_train_v2.sh local_configs/sardet100k_tfa/split2/tfa_sardet100k_split2_2xb8_BT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/tfa_sardet100k_split2_2xb8_BT/latest.pth --method random_init --save-dir work_dirs/tfa_sardet100k_split2_2xb8_BT/ --sardet100k

# Step 2: Fine-tuning (6 shots)
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sardet100k_GFSDet/split2/GFSDet_sardet100k_split2_2xb4_${shot}shot_FT.py"
    echo ">>> Running Task: sardet100k_GFSDet/split2 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
done

