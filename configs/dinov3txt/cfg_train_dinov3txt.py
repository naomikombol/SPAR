# model settings
model = dict(
    type='VitForSegmentation',
    wrapper = 'DINOv3txtWrapper',
    model = 'dinov3_vitl16_dinotxt_tet1280d20h24l',
    weights = "YOUR_DIRECTORY/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    backbone_weights = "YOUR_DIRECTORY/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    git_name = "facebookresearch/dinov3",
    slide_crop = 384,
    slide_stride =  0, 
    slide_stride_gen = 24,

    gen_windowslide_bs = 10,
    eval_windowslide_bs = 60,
)

processing_params = dict(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
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
    gt_dir = "YOUR_DIRECTORY/dinov3txt_embeddings/",
)


