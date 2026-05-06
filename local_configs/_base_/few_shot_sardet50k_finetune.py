# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=[0.3, 0.3, 0.1], direction=['horizontal', 'vertical','diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotCocoDataset
data_root = '/data/SOI_Det/SARDet_50K/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        save_dataset=True,
        type='FewShotSARDet50KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'Annotations/train.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes='ALL_CLASSES',
        instance_wise=False),
    val=dict(
        type='FewShotSARDet50KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'Annotations/val.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        pipeline=test_pipeline,
        classes='ALL_CLASSES'),
    test=dict(
        type='FewShotSARDet50KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'Annotations/val.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES'))
evaluation = dict(
    interval=4000,
    metric='bbox',
    classwise=True,
    class_splits=['BASE_CLASSES', 'NOVEL_CLASSES'])
