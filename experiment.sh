# sardet50k experiment


# TFA experiment
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb8_BT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/tfa_sardet50k_split1_2xb8_BT/iter_34644.pth --method random_init --save-dir work_dirs/tfa_sardet50k_split1_2xb8_BT/ --sardet50k
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb16_1shot_FT.py 2
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb16_2shot_FT.py 2
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb16_3shot_FT.py 2
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb2_5shot_FT.py 2
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb8_10shot_FT.py 2
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb2_30shot_FT.py 2
# python tools/detection/test.py local_configs/sardet50k_tfa/tfa_sardet50k_split1_2xb16_30shot_FT.py work_dirs/tfa_sardet50k_split1_2xb16_30shot_FT/latest.pth --eval bbox

# MPSR experiment
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_mpsr/mpsr_sardet50k_split1_2xb4_BT.py 2
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_mpsr/mpsr_sardet50k_split1_2xb8_30shot_FT.py 2

# python tools/detection/test.py local_configs/sardet50k_mpsr/mpsr_sardet50k_split1_2xb4_30shot_FT.py work_dirs/mpsr_sardet50k_split1_2xb4_30shot_FT/latest.pth --eval bbox




# sardet100k benchmark
# TFA experiment
# bash tools/detection/dist_train_v2.sh local_configs/sardet100k_tfa/tfa_sardet100k_split1_4xb4_BT.py 4
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/tfa_sardet100k_split1_4xb4_BT/iter_110000.pth --method random_init --save-dir work_dirs/tfa_sardet100k_split1_4xb4_BT/ --sardet100k
# for shot in 1 2 3 5 10 30
# do
#     CONFIG="local_configs/sardet100k_tfa/tfa_sardet100k_split1_2xb8_${shot}shot_FT.py"
    
#     echo ">>> Running Task:TFA sardet100k SHOT_NUM=${shot} with ${CONFIG}..."
#     SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
# done
# python tools/detection/test.py local_configs/sardet100k_tfa/tfa_sardet100k_split1_2xb2_10shot_FT.py work_dirs/tfa_sardet100k_split1_2xb2_10shot_FT/latest.pth --eval bbox

# FSCE experiment
# for shot in 1 2 3 5 10 30
# do
#     CONFIG="local_configs/sardet100k_fsce/fsce_sardet100k_split1_2xb8_${shot}shot_FT.py"
    
#     echo ">>> Running Task:MPSR sardet100k SHOT_NUM=${shot} with ${CONFIG}..."
#     SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
# done

# MPSR experiment
# bash tools/detection/dist_train_v2.sh local_configs/sardet50k_mpsr/mpsr_sardet50k_split1_2xb4_BT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/mpsr_sardet100k_split1_4xb4_BT/iter_80000.pth --method random_init --save-dir work_dirs/mpsr_sardet100k_split1_4xb4_BT/ --sardet100k
# for shot in 1 2 3 5 10 30
# do
#     CONFIG="local_configs/sardet100k_mpsr/mpsr_sardet100k_split1_2xb2_${shot}shot_FT.py"
    
#     echo ">>> Running Task:MPSR sardet100k SHOT_NUM=${shot} with ${CONFIG}..."
#     SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
# done

# GFSDet experiment
# for shot in 30
# do
#     CONFIG="local_configs/sardet100k_GFSDet/GFSDet_sardet100k_split1_2xb4_${shot}shot_FT.py"
    
#     echo ">>> Running Task:GFSDet sardet100k SHOT_NUM=${shot} with ${CONFIG}..."
#     SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
# done

# meta faster rcnn experiment
# bash tools/detection/dist_train_v2.sh \
#     local_configs/sardet100k_meta_faster_rcnn/meta_faster_rcnn_sardet100k_split1_2xb8_1shot_FT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/meta-rcnn_sardet100k_split1_4xb8_BT/latest.pth --method random_init --save-dir work_dirs/meta-rcnn_sardet100k_split1_4xb8_BT/ --sardet100k
# --- 任务: meta-rcnn ---
# for shot in 10 30
# do
#     CONFIG="local_configs/sardet100k_meta-rcnn/meta-rcnn_sardet100k_split1_2xb16-split1_${shot}shot_FT.py"
    
#     echo ">>> Running Task:meta-rcnn sardet100k SHOT_NUM=${shot} with ${CONFIG}..."
#     SHOT_NUM=$shot bash tools/detection/dist_train_v2.sh "$CONFIG" 2
# done


# sar-aircraft-1.0 experiment
# TFA experiment
bash tools/detection/dist_train_v2.sh local_configs/sar-aircraft_tfa/tfa_sar-aircraft_split1_2xb2_BT.py 2
python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/tfa_sar-aircraft_split1_2xb2_BT/latest.pth --method random_init --save-dir work_dirs/tfa_sar-aircraft_split1_2xb2_BT/ --sar_aircraft

# MPSR experiment
# bash tools/detection/dist_train_v2.sh local_configs/sar-aircraft_mpsr/mpsr_sar-aircraft_split1_2xb2_BT.py 2
# python tools/detection/misc/initialize_bbox_head.py --src1 work_dirs/mpsr_sar-aircraft_split1_2xb2_BT/latest.pth --method random_init --save-dir work_dirs/mpsr_sar-aircraft_split1_2xb2_BT/ --sar_aircraft
