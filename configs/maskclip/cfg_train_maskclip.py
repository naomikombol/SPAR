# model settings
model = dict(
    type='VitForSegmentation',
    wrapper = 'OpenCLIPWrapper',
    pretrained = 'laion2b_s34b_b88k',
    clip_path='ViT-B-16',
    slide_crop = 224,
    slide_stride =  0,
    slide_stride_gen = 24,

    gen_windowslide_bs = 10,
    eval_windowslide_bs = 60,
)

processing_params = dict(
    mean = [0.48145466, 0.4578275, 0.40821073],
    std = [0.26862954, 0.26130258, 0.27577711],
)

training_params = dict(
    gen_num_epochs = 1,
    train_num_epochs = 10,

    learning_rate = 2e-5,
    weight_decay = 1e-4,

    batch_size = 1, 
    num_workers = 15,

    train_devices =  2,
    gen_devices =  1,
    
    precision = 16,
    accelerator = "gpu",
    strategy = "ddp_find_unused_parameters_true" ,

    unfrozen_config = "blk_-1,blk_-2",
    
    img_dir = "YOUR_DIRECTORY/images/",
    gt_dir = "YOUR_DIRECTORY/maskclip_embeddings/",
)


