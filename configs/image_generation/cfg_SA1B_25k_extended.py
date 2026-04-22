_base_ = './cfg_SA1B_25k.py'

generation_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResize', scale=(2560, 2560),ratio_range=(0.2, 1.0), keep_ratio = True), 
    dict(type='RandomCropOrFullImage', prob_crop=0.5), 
    dict(type='ImageToTensor', keys=['img']),
    dict(type='RandomHorizontalFlipTensor', prob=0.5),
]
