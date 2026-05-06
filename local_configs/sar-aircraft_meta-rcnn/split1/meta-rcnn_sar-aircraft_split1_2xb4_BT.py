_base_ = [
    '../../_base_/base_sar-aircraft_nwaykshot.py',
    '../../_base_/schedule.py',
    '../meta-rcnn_r101_c4.py',
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

total_images = 14221  # base_trainval.json: 14221张（split1基类5类）
total_epoch = 12
step_epoch = 8
batch_size = 4
gpu_number = 2
num_classes = 5

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=num_classes, num_meta_classes=num_classes)))

lr_config = dict(warmup_iters=500, 
                 step=[total_images//(batch_size*gpu_number) * step_epoch,
        total_images//(batch_size*gpu_number) * (total_epoch-1)])
# auto_scale_lr = dict(base_batch_size=2*8, enable=True)
runner = dict(max_iters=total_images//(batch_size*gpu_number) * total_epoch)
# model settings
data = dict(samples_per_gpu=batch_size)
evaluation = dict(interval=total_images//(batch_size*gpu_number)*2,
                  metric='bbox',
                  classwise=True)
checkpoint_config = dict(interval=total_images//(batch_size*gpu_number)*2)
optimizer = dict(lr=0.005)  # 2 gpu 8 batch size

# resume_from = 'work_dirs/tfa_sardet50k_2xb4_BT/latest.pth'