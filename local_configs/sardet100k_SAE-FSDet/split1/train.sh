#!/bin/bash
# sardet100k_SAE-FSDet split1 training script

# Step 1: Base Training
# bash tools/detection/dist_train_v2.sh local_configs/sardet100k_SAE-FSDet/split1/SAE-FSDet_sardet100k_split1_4xb2_BT.py 4

# Step 2: Fine-tuning
for shot in 10
do
    CONFIG="local_configs/sardet100k_SAE-FSDet/split1/SAE-FSDet_sardet100k_split1_4xb2_${shot}shot_FT.py"
    echo ">>> Running Task: sardet100k_SAE-FSDet/split1 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 4
done
