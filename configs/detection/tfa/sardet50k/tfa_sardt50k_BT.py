_base_ = [
    '../../_base_/datasets/fine_tune_based/base_sardet50k.py',
    '../../_base_/schedules/schedule.py',
    '../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]

# # optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[60000, 80000])
# runner = dict(type='IterBasedRunner', max_iters=90000)

total_images = 94493
total_epoch = 12
step_epoch = 8
batch_size = 8
gpu_number = 2
num_classes = 5

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=5)))

lr_config = dict(warmup_iters=1000, 
                 step=[total_images//(batch_size*gpu_number) * step_epoch,
        total_images//(batch_size*gpu_number) * (total_epoch-1)])
auto_scale_lr = dict(base_batch_size=2*8, enable=True)
runner = dict(max_iters=total_images//(batch_size*gpu_number) * total_epoch)
# model settings

evaluation = dict(interval=total_images//(batch_size*gpu_number),
                  metric='bbox',
                  classwise=True)
