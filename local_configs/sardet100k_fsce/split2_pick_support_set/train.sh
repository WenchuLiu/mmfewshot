#!/bin/bash
# Train candidate support sets, then select by validation metrics.
#
# Examples:
#   bash local_configs/sardet100k_fsce/split2_pick_support_set/train.sh
#   SHOT=1 bash local_configs/sardet100k_fsce/split2_pick_support_set/train.sh
#   SHOT=3 SEEDS="0 1 2 3 4" bash local_configs/sardet100k_fsce/split2_pick_support_set/train.sh

SHOT=${SHOT:-3}
SEEDS=${SEEDS:-"0 1 2 3 4"}

for seed in ${SEEDS}
do
    CONFIG="local_configs/sardet100k_fsce/split2_pick_support_set/fsce_sardet100k_split2_2xb8_${SHOT}shot_seed${seed}_FT.py"
    echo ">>> Running Task: sardet100k_fsce/split2_pick_support_set ${SHOT}shot seed=${seed} ..."
    bash tools/detection/dist_train_v2.sh "$CONFIG" 2
done
