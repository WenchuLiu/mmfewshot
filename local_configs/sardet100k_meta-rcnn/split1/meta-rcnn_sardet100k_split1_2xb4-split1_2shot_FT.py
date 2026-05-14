_base_ = [
    '../../_base_/few_shot_sardet100k_nwaykshot.py',
    '../../_base_/schedule.py', '../meta-rcnn_r101_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
num_shots = 2
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotSARDet100KDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting=f'SPLIT1_{num_shots}SHOT')],
            num_novel_shots=num_shots,
            num_base_shots=num_shots,
            )),
        model_init=dict(classes='ALL_CLASSES'))
evaluation = dict(
    interval=100, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=100)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=300)
load_from = \
    'work_dirs/meta-rcnn/sardet100k/split1/2xb4_BT/base_model_random_init_bbox_head.pth'
work_dir = 'work_dirs/meta-rcnn/sardet100k/split1/2xb4-split1_2shot_FT/'
# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
])
