_base_ = [
    '../../_base_/base_sardet100k_finetune.py',
    '../../_base_/schedule.py',
    '../../_base_/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]

split_id = 2
batch_size = 4
num_classes = 5

# Override classes and ann paths for split2.
rpn_weight = 0.7

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    rpn_head=dict(
        _delete_=True,
        type='GradualRPNHead',
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
    roi_head=dict(
        bbox_head=dict(reg_class_agnostic=True, num_classes=num_classes)))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=18)
data = dict(samples_per_gpu=batch_size, workers_per_gpu=4,
    train=dict(
        classes='BASE_CLASSES_SPLIT2',
        ann_cfg=[dict(type='ann_file', ann_file='data/sardet100k/split2/base_train.json')]),
    val=dict(
        classes='BASE_CLASSES_SPLIT2',
        ann_cfg=[dict(type='ann_file', ann_file='data/sardet100k/split2/base_test.json')],
        img_prefix='data/sardet100k/JPEGImages/test/'),
    test=dict(
        classes='BASE_CLASSES_SPLIT2',
        ann_cfg=[dict(type='ann_file', ann_file='data/sardet100k/split2/FewShot_test.json')],
        img_prefix='data/sardet100k/JPEGImages/test/'))
evaluation = dict(
    interval=6,
    metric='bbox',
    classwise=True)
checkpoint_config = dict(interval=2)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

work_dir = 'work_dirs/SAE-FSDet/sardet100k/split2/4xb4_BT/'
