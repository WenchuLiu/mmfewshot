_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco.py',
    '../../_base_/schedules/schedule.py',
    '../vfa_r101_c4.py',
    '../../_base_/default_runtime.py'
]

data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=10,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
        )),
    model_init=dict(num_novel_shots=10, num_base_shots=10))
evaluation = dict(interval=10000)
checkpoint_config = dict(interval=10000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[10000])
runner = dict(max_iters=10000)
load_from = 'work_dirs/vfa_r101_c4_8xb4_coco_base-training/latest.pth'

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80, num_meta_classes=80)),
    with_refine=True,
    frozen_parameters=[
        'backbone', 'shared_head', 'aggregation_layer', 'rpn_head.rpn_conv'
    ])
