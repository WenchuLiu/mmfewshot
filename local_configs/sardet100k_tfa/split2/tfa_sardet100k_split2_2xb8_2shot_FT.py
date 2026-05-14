_base_ = [
    '../../_base_/few_shot_sardet100k_finetune.py',
    '../../_base_/schedule.py',
    '../tfa_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset

batch_size = 4
num_gpus = 2
num_classes = 6
num_shots = 2

# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type='FewShotSARDet100KDefaultDataset',
        ann_cfg=[dict(method='TFA', setting=f'SPLIT2_{num_shots}SHOT')],
        num_novel_shots=num_shots,
        num_base_shots=num_shots,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))

# 几shot就 1200*几 个iter，因为coco有80个类，sardet50k一共有6个类，
# 80个类有160000个iter，对应sardet50k就是12000个iter，对应sardet50k就是每个 shot 1200个iter
# 原来设置是2xb2，针对batchsize不同要做更改
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=100, gamma=0.3, step=[4500])
runner = dict(max_iters=6000)
model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes)))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

# auto_scale_lr没有起作用
auto_scale_lr = dict(base_batch_size=2*2, enable=True)

load_from = ('work_dirs/tfa/sardet100k/split2/2xb8_BT/'
             'base_model_random_init_bbox_head.pth')
work_dir = 'work_dirs/tfa/sardet100k/split2/2xb8_2shot_FT/'
