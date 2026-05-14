_base_ = [
    '../../_base_/base_sar-aircraft_twobranch.py',
    '../../_base_/schedule.py', 
    '../mpsr_r101_fpn.py',
    '../../_base_/default_runtime.py'
]

total_images = 13386  # base_trainval.json: 13386张（split2基类5类）
total_epoch = 12
step_epoch = 8
batch_size = 4
gpu_number = 2
num_classes = 5

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes)))

optimizer = dict(
    lr=0.005/2,
    paramwise_cfg=dict(
        custom_keys=dict({'.bias': dict(lr_mult=2.0, decay_mult=0.0)})))
lr_config = dict(warmup_iters=500, 
                 warmup_ratio=1. / 3,
                 step=[total_images//(batch_size*gpu_number) * step_epoch,
        total_images//(batch_size*gpu_number) * (total_epoch-1)])
# auto_scale_lr = dict(base_batch_size=2*8, enable=True)

runner = dict(max_iters=total_images//(batch_size*gpu_number) * total_epoch)

evaluation = dict(interval=total_images//(batch_size*gpu_number)*2,
                  metric='bbox',
                  classwise=True)

# debug
# evaluation = dict(interval=100,
#                   metric='bbox',
#                   classwise=True)

checkpoint_config = dict(interval=total_images//(batch_size*gpu_number)*2)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    auxiliary_samples_per_gpu=batch_size,
    auxiliary_workers_per_gpu=batch_size,
    train=dict(
        dataset=dict(classes='BASE_CLASSES_SPLIT2')),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'))

resume_from = './work_dirs/mpsr/sar-aircraft/split2/2xb8_BT/iter_6654.pth'
work_dir = 'work_dirs/mpsr/sar-aircraft/split2/2xb2_BT/'
