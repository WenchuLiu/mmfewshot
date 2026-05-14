#!/bin/bash
# sardet100k_fsce split2 training script

# Prerequisite: TFA Base Training (if not already done)
# bash tools/detection/dist_train_v2.sh local_configs/sardet100k_tfa/split2/tfa_sardet100k_split2_2xb8_BT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/tfa/sardet100k/split2/2xb8_BT/latest.pth --method random_init --save-dir work_dirs/tfa/sardet100k/split2/2xb8_BT/ --sardet100k --sardet100k_split 2

# Step 2: Fine-tuning (6 shots)
for shot in 5
do
    CONFIG="local_configs/sardet100k_fsce/split2/fsce_sardet100k_split2_2xb8_${shot}shot_FT.py"
    echo ">>> Running Task: sardet100k_fsce/split2 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
done

