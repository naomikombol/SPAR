# Where the images will be saved after augmentation
raw_img_dir = "YOUR_DIRECTORY/"
img_dir = "YOUR_DIRECTORY/images/"

generation_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResize', scale=(2048,1024),ratio_range=(0.5, 1.0), keep_ratio = True), 
    dict(type='RandomCropOrFullImage', prob_crop=0.5),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='RandomHorizontalFlipTensor', prob=0.5),
]

generation_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ImageOnlyDataset',
        data_root= raw_img_dir,
        img_subdir='raw_images', 
        pipeline=generation_pipeline,
    )
)
