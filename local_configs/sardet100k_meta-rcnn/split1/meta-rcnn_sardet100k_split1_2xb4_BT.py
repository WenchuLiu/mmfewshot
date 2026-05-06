_base_ = [
    '../../_base_/base_sardet100k_nwaykshot.py',
    '../../_base_/schedule.py',
    '../meta-rcnn_r101_c4.py',
    '../../_base_/default_runtime.py'
]

total_images = 79625
total_epoch = 12
step_epoch = 8
batch_size = 4
gpu_number = 2
num_classes = 5

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=num_classes)))

lr_config = dict(warmup_iters=500,
                 step=[total_images//(batch_size*gpu_number) * step_epoch,
        total_images//(batch_size*gpu_number) * (total_epoch-1)])
runner = dict(max_iters=total_images//(batch_size*gpu_number) * total_epoch)

evaluation = dict(interval=total_images//(batch_size*gpu_number)*6,
                  metric='bbox',
                  classwise=True)
checkpoint_config = dict(interval=total_images//(batch_size*gpu_number))
optimizer = dict(lr=0.02 * batch_size * gpu_number / 16)

auto_scale_lr = dict(base_batch_size=2*8, enable=True)
