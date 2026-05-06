_base_ = [
    '../../_base_/few_shot_sardet100k_finetune.py',
    # '../../_base_/schedule.py', '../fsce_r101_fpn_contrastive_loss.py',
    '../../_base_/schedule.py', '../fsce_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
num_shots = 5
num_classes = 6

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='FewShotSARDet100KDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting=f'{num_shots}SHOT')],
        num_novel_shots=num_shots,
        num_base_shots=num_shots,
        classes='ALL_CLASSES_SPLIT2'))


# evaluation = dict(interval=5000)
# checkpoint_config = dict(interval=5000)
# optimizer = dict(lr=0.001)
# lr_config = dict(warmup_iters=200, gamma=0.3, step=[20000])
# runner = dict(max_iters=30000)
# model = dict(
#     roi_head=dict(bbox_head=dict(num_classes=80)),
#     train_cfg=dict(
#         rcnn=dict(
#             assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5))))


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


load_from = ('work_dirs/tfa_sardet100k_split2_2xb8_BT/'
             'base_model_random_init_bbox_head.pth')
