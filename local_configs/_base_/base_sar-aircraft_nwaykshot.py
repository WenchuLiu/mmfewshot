# dataset settings for Meta-RCNN base training on SAR-Aircraft-1.0
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = 'data/SAR-Aircraft-1.0/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='NWayKShotDataset',
        num_support_ways=5,
        num_support_shots=1,
        one_support_shot_per_image=False,
        num_used_support_shots=30,
        save_dataset=False,
        dataset=dict(
            type='FewShotSARAircraftDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                ann_file=data_root + 'split1/base_trainval.json')
            ],
            img_prefix=data_root+'JPEGImages/',
            multi_pipelines=train_multi_pipelines,
            classes='BASE_CLASSES_SPLIT1',
            instance_wise=False,
            dataset_name='query_support_dataset')),
    val=dict(
        type='FewShotSARAircraftDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/base_test.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        pipeline=test_pipeline,
        classes='BASE_CLASSES_SPLIT1'),
    test=dict(
        type='FewShotSARAircraftDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'split1/ft_test.json')
        ],
        img_prefix=data_root+'JPEGImages/',
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES_SPLIT1'),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotSARAircraftDataset',
        ann_cfg=None,
        img_prefix= data_root+'JPEGImages/',
        pipeline=train_multi_pipelines['support'],
        instance_wise=True,
        classes='BASE_CLASSES_SPLIT1',
        dataset_name='model_init_dataset'))
evaluation = dict(interval=3000, metric='bbox', class_splits=None)
