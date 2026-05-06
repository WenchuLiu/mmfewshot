# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
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
# classes splits are predefined in FewShotSARAircraftDataset
data_root = 'data/SAR-Aircraft-1.0/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        save_dataset=True,
        type='FewShotSARAircraftDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'Annotations/trainval.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes='ALL_CLASSES_SPLIT1',
        instance_wise=False),
    val=dict(
        type='FewShotSARAircraftDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'Annotations/test.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        pipeline=test_pipeline,
        classes='ALL_CLASSES_SPLIT1'),
    test=dict(
        type='FewShotSARAircraftDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'Annotations/test.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=4000,
    metric='bbox',
    classwise=True,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
