_base_ = [
    '../../_base_/few_shot_sar-aircraft_finetune.py',
    '../../_base_/schedule.py',
    '../tfa_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotSARAircraftDataset

num_classes = 7
num_shots = 30

# FewShotSARAircraftDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='FewShotSARAircraftDefaultDataset',
        ann_cfg=[dict(method='TFA', setting=f'{num_shots}SHOT')],
        num_novel_shots=num_shots,
        num_base_shots=num_shots,
        classes='ALL_CLASSES_SPLIT2'))

evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=100, gamma=0.3, step=[4500])
runner = dict(max_iters=6000)
model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes)))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py

load_from = ('work_dirs/tfa_sar-aircraft_split2_2xb8_BT/'
             'base_model_random_init_bbox_head.pth')
