_base_ = [
    '../sardet100k_meta-rcnn/meta-rcnn_r50_c4.py',
]

pretrained = 'open-mmlab://detectron2/resnet101_caffe'

model = dict(
    type='VFA',
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='VFARoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='VFABBoxHead',
            num_classes=6,
            num_meta_classes=6)))
