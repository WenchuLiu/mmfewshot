_base_ = [
    '../../_base_/few_shot_sar-aircraft_finetune.py',
    '../../_base_/schedule.py', '../fsce_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotSARAircraftDataset
num_shots = 3
num_classes = 7

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='FewShotSARAircraftDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting=f'SPLIT2_{num_shots}SHOT')],
        num_novel_shots=num_shots,
        num_base_shots=num_shots))

evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=100, gamma=0.3, step=[4500])
runner = dict(max_iters=6000)
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=num_classes)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5))))

load_from = ('work_dirs/tfa/sar-aircraft/split2/2xb8_BT/'
             'base_model_random_init_bbox_head.pth')
work_dir = 'work_dirs/fsce/sar-aircraft/split2/2xb8_3shot_FT/'
