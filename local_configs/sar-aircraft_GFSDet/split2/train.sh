#!/bin/bash
# sar-aircraft_GFSDet split2 training script

# Prerequisite: TFA Base Training (if not already done)
# bash tools/detection/dist_train_v2.sh local_configs/sar-aircraft_tfa/split2/tfa_sar-aircraft_split2_2xb2_BT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/tfa/sar-aircraft/split2/2xb2_BT/latest.pth --method random_init --save-dir work_dirs/tfa/sar-aircraft/split2/2xb2_BT/ --sar_aircraft --sar_aircraft_split 2

# Step 2: Fine-tuning (6 shots)
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sar-aircraft_GFSDet/split2/GFSDet_sar-aircraft_split2_2xb2_${shot}shot_FT.py"
    echo ">>> Running Task: sar-aircraft_GFSDet/split2 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
done

