#!/bin/bash
# sar-aircraft_vfa split2 training script

# Step 1: Base Training
bash tools/detection/dist_train_v2.sh local_configs/sar-aircraft_vfa/split2/vfa_sar-aircraft_split2_4xb4_BT.py 2

# Step 2: Initialize bbox head, bbox regressor, and meta classifier for all classes
python tools/detection/misc/initialize_bbox_head.py \
    --src1 work_dirs/vfa/sar-aircraft/split2/4xb4_BT/latest.pth \
    --method random_init \
    --param-name roi_head.bbox_head.fc_cls roi_head.bbox_head.fc_reg roi_head.bbox_head.fc_meta \
    --save-dir work_dirs/vfa/sar-aircraft/split2/4xb4_BT/ \
    --sar_aircraft --sar_aircraft_split 2

# Step 3: Fine-tuning
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sar-aircraft_vfa/split2/vfa_sar-aircraft_split2_4xb4_${shot}shot_FT.py"
    echo ">>> Running Task: sar-aircraft_vfa/split2 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
done
