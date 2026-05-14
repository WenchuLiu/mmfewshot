_base_ = [
    '../../_base_/few_shot_sardet100k_finetune.py',
    '../../_base_/schedule.py',
    '../fsce_r101_fpn.py',
    '../../_base_/default_runtime.py'
]

batch_size = 2
num_classes = 6
num_shots = 3
seed = 2

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type='FewShotSARDet100KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/sardet100k/split2_pick_support_set/'
                f'FewShot_{num_shots}shot_train_seed{seed}.json')
        ],
        num_novel_shots=num_shots,
        num_base_shots=num_shots,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/sardet100k/split2/FewShot_test.json')
        ],
        img_prefix='data/sardet100k/JPEGImages/test/',
        classes='ALL_CLASSES_SPLIT2'),
    test=dict(
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/sardet100k/split2/FewShot_test.json')
        ],
        img_prefix='data/sardet100k/JPEGImages/test/',
        classes='ALL_CLASSES_SPLIT2'))

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

load_from = ('work_dirs/tfa/sardet100k/split2/2xb8_BT/'
             'base_model_random_init_bbox_head.pth')
work_dir = (
    'work_dirs/fsce/sardet100k/split2_pick_support_set/'
    f'2xb8_{num_shots}shot_seed{seed}_FT/')
