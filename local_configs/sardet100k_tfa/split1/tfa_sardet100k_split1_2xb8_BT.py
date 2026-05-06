_base_ = [
    '../../_base_/base_sardet100k_finetune.py',
    '../../_base_/schedule.py',
    '../../_base_/faster_rcnn_r50_caffe_fpn.py',
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

total_images = 79625
total_epoch = 12
step_epoch = 8
batch_size = 8
gpu_number = 2
num_classes = 5

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=5)))

lr_config = dict(warmup_iters=500, 
                 step=[total_images//(batch_size*gpu_number) * step_epoch,
        total_images//(batch_size*gpu_number) * (total_epoch-1)])
runner = dict(max_iters=total_images//(batch_size*gpu_number) * total_epoch)
# model settings
data = dict(samples_per_gpu=batch_size,
            workers_per_gpu=batch_size)
evaluation = dict(interval=total_images//(batch_size*gpu_number)*6,
                  metric='bbox',
                  classwise=True)
checkpoint_config = dict(interval=total_images//(batch_size*gpu_number))
optimizer = dict(lr=0.02 * batch_size * gpu_number / 16)


auto_scale_lr = dict(base_batch_size=2*8, enable=True)
resume_from = \
    'work_dirs/tfa_sardet100k_split1_2xb8_BT/latest.pth'