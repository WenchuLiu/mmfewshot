_base_ = [
    '../../_base_/few_shot_sardet100k_finetune.py',
    '../../_base_/schedule.py', '../tfa_r101_fpn.py',
    '../../_base_/default_runtime.py'
]

num_shots = 3
num_classes = 6

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='FewShotSARDet100KDefaultDataset',
        ann_cfg=[dict(method='GFSDet', setting=f'SPLIT1_{num_shots}SHOT')],
        num_novel_shots=num_shots,
        num_base_shots=num_shots))

evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
optimizer = dict(lr=0.001)
optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(warmup_iters=100, step=[4500])
runner = dict(max_iters=6000)

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    frozen_parameters=[
        'backbone', 'neck', 'roi_head.bbox_head.base_shared_fcs',],

    roi_head=dict(
        bbox_head=dict(
            type='DisKDBBoxHead',
            num_classes=num_classes,
            loss_kd_weight= 0.025,
            base_alpha=0.5,
            loss_bbox=dict(loss_weight=2.0),
            loss_cls=dict(loss_weight=1.0),
            dis_loss=dict(
                type='DisLoss', num_classes=num_classes, shot=num_shots, 
                loss_base_margin_weight=1.0,
                loss_novel_margin_weight=1.0,
                loss_neg_margin_weight=1.0,
                power_weight=4.0,
                novel_class_ids=novel_class_ids),
            base_cpt = 'work_dirs/tfa/sardet100k/split1/2xb8_BT/base_model_random_init_bbox_head.pth',
            novel_class_ids = [1],
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ]
            )))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

load_from = ('work_dirs/tfa/sardet100k/split1/2xb8_BT/'
             'base_model_random_init_bbox_head.pth')
work_dir = 'work_dirs/GFSDet/sardet100k/split1/2xb4_3shot_FT/'
