#!/bin/bash
# sardet100k_meta-rcnn split1 training script

# Step 1: Base Training
bash tools/detection/dist_train_v2.sh local_configs/sardet100k_meta-rcnn/split1/meta-rcnn_sardet100k_split1_2xb4_BT.py 2

# Step 2: Initialize bbox head
python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/meta-rcnn/sardet100k/split1/2xb4_BT/latest.pth --method random_init --save-dir work_dirs/meta-rcnn/sardet100k/split1/2xb4_BT/ --sardet100k --sardet100k_split 1

# Step 3: Fine-tuning (6 shots)
for shot in 1 2 3 5 10 30
do
    CONFIG="local_configs/sardet100k_meta-rcnn/split1/meta-rcnn_sardet100k_split1_2xb4-split1_${shot}shot_FT.py"
    echo ">>> Running Task: sardet100k_meta-rcnn/split1 SHOT_NUM=${shot} ..."
    SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
done

