_base_ = [
    '../../_base_/few_shot_sardet100k_twobranch.py',
    '../../_base_/schedule.py', '../mpsr_r101_fpn.py',
    '../../_base_/default_runtime.py'
]

batch_size = 2
num_gpus = 2
num_classes = 6
num_shots = 10

# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    auxiliary_samples_per_gpu=batch_size,
    auxiliary_workers_per_gpu=batch_size,
    train=dict(
        dataset=dict(
            type='FewShotSARDet100KDefaultDataset',
            ann_cfg=[dict(method='MPSR', setting=f'{num_shots}SHOT')],
            num_novel_shots=num_shots,
            num_base_shots=num_shots,
            classes='ALL_CLASSES_SPLIT2')))

# 原来设置是2xb2，针对batchsize不同要做更改
# evaluation默认2k一次，一共4k个iter训练，2xb2的设置
# 原来设置是2xb2，针对batchsize不同要做更改
base_batch_size = 4  # 原始配置对应的batch size
total_batch_size = batch_size * num_gpus
scale_factor = total_batch_size / base_batch_size

# 学习率线性缩放
optimizer = dict(lr=0.005 * scale_factor)

# 训练步数反比缩放
max_iters = max(100, int(2000 / scale_factor))

# 其他参数与max_iters成比例
evaluation = dict(interval=max(10, max_iters))
checkpoint_config = dict(interval=max_iters)
lr_config = dict(
    warmup_iters=max(50, int(max_iters * 0.05)),
    warmup_ratio=1. / 3,
    step=[int(max_iters * 0.7)]
)
runner = dict(max_iters=max_iters)

load_from = \
    'work_dirs/mpsr_sardet100k_split2_2xb2_BT/base_model_random_init_bbox_head.pth'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
            init_cfg=[
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.001))
            ])))
