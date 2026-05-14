_base_ = [
    '../../_base_/few_shot_sardet100k_nwaykshot.py',
    '../../_base_/schedule.py',
    '../vfa_r101_c4.py',
    '../../_base_/default_runtime.py'
]

num_shots = 5
num_classes = 6

data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotSARDet100KDefaultDataset',
            ann_cfg=[dict(method='VFA', setting=f'SPLIT1_{num_shots}SHOT')],
            num_novel_shots=num_shots,
            num_base_shots=num_shots)),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=1000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=200)
optimizer = dict(lr=0.0005)
lr_config = dict(warmup=None)
runner = dict(max_iters=800)
load_from =     'work_dirs/vfa/sardet100k/split1/4xb4_BT/base_model_random_init_bbox_head.pth'
work_dir = 'work_dirs/vfa/sardet100k/split1/4xb4_5shot_FT/'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
            num_meta_classes=num_classes)),
    with_refine=False,
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ])
