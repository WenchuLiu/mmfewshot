#!/bin/bash
# sar-aircraft_SAE-FSDet split2 training script

# Step 1: Base Training
bash tools/detection/dist_train_v2.sh local_configs/sar-aircraft_SAE-FSDet/split2/SAE-FSDet_sar-aircraft_split2_4xb2_BT.py 4

# Step 2: Fine-tuning
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sar-aircraft_SAE-FSDet/split2/SAE-FSDet_sar-aircraft_split2_4xb2_${shot}shot_FT.py"
    echo ">>> Running Task: sar-aircraft_SAE-FSDet/split2 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 4
done
