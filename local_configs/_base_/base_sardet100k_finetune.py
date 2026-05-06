# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    # dict(type='RandomRotate90', prob=1.0),
    # dict(type='RandomFlip', flip_ratio=[0.3, 0.3, 0.1], direction=['horizontal', 'vertical','diagonal']),
    dict(type='RandomFlip', flip_ratio=0.5),
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
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='FewShotSARDet100KDataset',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/base_train.json')
        ],
        img_prefix=data_root+'JPEGImages/train/',
        pipeline=train_pipeline,
        classes='BASE_CLASSES_SPLIT1'),
    val=dict(
        type='FewShotSARDet100KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file = data_root + 'Annotations/val.json')
        ],
        img_prefix=data_root+'JPEGImages/val/',
        pipeline=test_pipeline,
        classes='BASE_CLASSES_SPLIT1'),
    test=dict(
        type='FewShotSARDet100KDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/FewShot_test.json')
        ],
        img_prefix=data_root+'JPEGImages/val/',
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES_SPLIT1'))
evaluation = dict(interval=5000, metric='bbox', classwise=True)
