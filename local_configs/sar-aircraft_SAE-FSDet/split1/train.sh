#!/bin/bash
# sar-aircraft_SAE-FSDet split1 training script

# Step 1: Base Training
bash tools/detection/dist_train_v2.sh local_configs/sar-aircraft_SAE-FSDet/split1/SAE-FSDet_sar-aircraft_split1_4xb2_BT.py 4

# Step 2: Initialize bbox head and bbox regressor for all classes
python tools/detection/misc/initialize_bbox_head.py \
    --src1 work_dirs/SAE-FSDet/sar-aircraft/split1/4xb2_BT/latest.pth \
    --method random_init \
    --param-name roi_head.bbox_head.fc_cls roi_head.bbox_head.fc_reg \
    --save-dir work_dirs/SAE-FSDet/sar-aircraft/split1/4xb2_BT/ \
    --sar_aircraft --sar_aircraft_split 1

# Step 3: Fine-tuning
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sar-aircraft_SAE-FSDet/split1/SAE-FSDet_sar-aircraft_split1_4xb2_${shot}shot_FT.py"
    echo ">>> Running Task: sar-aircraft_SAE-FSDet/split1 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 4
done
