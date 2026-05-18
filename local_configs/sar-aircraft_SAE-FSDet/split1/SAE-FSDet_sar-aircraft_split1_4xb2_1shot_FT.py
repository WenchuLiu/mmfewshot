_base_ = [
    '../../_base_/few_shot_sar-aircraft_finetune.py',
    '../../_base_/schedule.py',
    '../../_base_/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]

split_id = 1
num_shots = 1
num_base_classes = 5
num_novel_classes = 2
num_classes = 7

classes = ('Boeing737', 'Boeing787', 'A330', 'A220', 'A320orA321', 'ARJ21', 'other')
base_classes = ('Boeing787', 'A330', 'A220', 'A320orA321', 'other')
novel_classes = ('Boeing737', 'ARJ21')
base_label_ids = tuple(i for i, c in enumerate(classes) if c in base_classes)
novel_label_ids = tuple(i for i, c in enumerate(classes) if c in novel_classes)

rpn_weight = 0.7

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    rpn_head=dict(
        _delete_=True,
        type='GradualRPNHead',
        aug_proposal=True,
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageGradualRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='dcn'),
                bridged_feature=False,
                with_cls=True,
                reg_decoded_bbox=True,
                point_generator=dict(
                    type='MlvlPointGenerator', strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ]),
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=0.3,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=0.3),
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0)))


model.update(
    frozen_parameters=['backbone'],
    roi_head=dict(
        type='LCCRoIHead',
        bbox_head=dict(
            _delete_=True,
            type='LCCBoxHead',
            use_dropout=False,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_shared_fcs=2,
            num_classes=num_base_classes,
            num_novel_classes=num_novel_classes,
            base_label_ids=base_label_ids,
            novel_label_ids=novel_label_ids,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='LCCFocalLoss',
                num_base_classes=num_base_classes,
                num_novel_classes=num_novel_classes,
                base_label_ids=base_label_ids,
                novel_label_ids=novel_label_ids,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ])))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotSARAircraftDefaultDataset',
        ann_cfg=[dict(method='SAE_FSDet', setting=f'SPLIT1_{num_shots}SHOT')],
        num_novel_shots=num_shots,
        num_base_shots=num_shots,
        classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
evaluation = dict(
    interval=108,
    metric='bbox',
    classwise=True,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=108)
log_config = dict(
    interval=1000, hooks=[dict(type='TextLoggerHook', ignore_last=False)])
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[190])
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=216)
load_from = 'work_dirs/SAE-FSDet/sar-aircraft/split1/4xb2_BT/latest.pth'
work_dir = 'work_dirs/SAE-FSDet/sar-aircraft/split1/4xb2_1shot_FT/'
