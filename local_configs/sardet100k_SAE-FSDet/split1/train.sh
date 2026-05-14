#!/bin/bash
# sardet100k_SAE-FSDet split1 training script

# Step 1: Base Training
bash tools/detection/dist_train_v2.sh local_configs/sardet100k_SAE-FSDet/split1/SAE-FSDet_sardet100k_split1_4xb4_BT.py 4

# Step 2: Fine-tuning
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sardet100k_SAE-FSDet/split1/SAE-FSDet_sardet100k_split1_4xb4_${shot}shot_FT.py"
    echo ">>> Running Task: sardet100k_SAE-FSDet/split1 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 4
done
