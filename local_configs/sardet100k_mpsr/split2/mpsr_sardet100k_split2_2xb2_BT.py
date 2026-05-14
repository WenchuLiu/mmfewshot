_base_ = [
    '../../_base_/few_shot_sardet100k_twobranch.py',
    '../../_base_/schedule.py',
    '../mpsr_r101_fpn.py',
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
batch_size = 2
gpu_number = 2
num_classes = 5

model = dict(
    
    roi_head=dict(bbox_head=dict(num_classes=num_classes)))

lr_config = dict(warmup_iters=500, 
                 step=[total_images//(batch_size*gpu_number) * step_epoch,
        total_images//(batch_size*gpu_number) * (total_epoch-1)])
runner = dict(max_iters=total_images//(batch_size*gpu_number) * total_epoch)
# model settings
data = dict(samples_per_gpu=batch_size,
            workers_per_gpu=batch_size,
            train=dict(
                classes='BASE_CLASSES_SPLIT2',
                dataset=dict(
                    ann_cfg=[dict(ann_file='data/sardet100k/split2/base_train.json')])),
            val=dict(
                classes='BASE_CLASSES_SPLIT2',
                ann_cfg=[dict(ann_file='data/sardet100k/split2/FewShot_test.json')]),
            test=dict(
                classes='BASE_CLASSES_SPLIT2',
                ann_cfg=[dict(ann_file='data/sardet100k/split2/FewShot_test.json')]))
evaluation = dict(interval=total_images//(batch_size*gpu_number)*6,
                  metric='bbox',
                  classwise=True)
checkpoint_config = dict(interval=total_images//(batch_size*gpu_number))
optimizer = dict(lr=0.02 * batch_size * gpu_number / 16)

resume_from = 'work_dirs/tfa/sardet100k/split2/2xb8_BT/latest.pth'

work_dir = 'work_dirs/mpsr/sardet100k/split2/2xb2_BT/'
