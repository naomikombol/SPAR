_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/siglip2/cls_ade20k.txt',
)


# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'YOUR_DIRECTORY/ADEChallengeData2016/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 672), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))