# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=[0.5,0.5], direction=['horizontal', 'vertical']),
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
data_root = 'data/sardet100k/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        save_dataset=True,
        type='FewShotSARDet100KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/base_train.json')
        ],
        img_prefix=data_root+'JPEGImages/train/',
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes='ALL_CLASSES_SPLIT1',
        instance_wise=False),
    val=dict(
        type='FewShotSARDet100KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/FewShot_test.json')
        ],
        img_prefix=data_root+'JPEGImages/test/',
        pipeline=test_pipeline,
        classes='ALL_CLASSES_SPLIT1'),
    test=dict(
        type='FewShotSARDet100KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/FewShot_test.json')
        ],
        img_prefix=data_root+'JPEGImages/test/',
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=4000,
    metric='bbox',
    classwise=True)
