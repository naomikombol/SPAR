# model settings
model_tokenizer_name = "hf-hub:timm/ViT-B-16-SigLIP2-512" 
model = dict(
    type='VitForSegmentation',
    wrapper = 'SigLIP2Wrapper',
    model = model_tokenizer_name, 
    tokenizer = model_tokenizer_name,
    slide_crop = int(model_tokenizer_name[-3:]),
    slide_stride =  0, 
    slide_stride_gen = 24,

    gen_windowslide_bs = 10,
    eval_windowslide_bs = 60,
)

processing_params = dict(
    mean = [0.5, 0.5, 0.5],
    std = [0.5, 0.5, 0.5],
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
    strategy = "ddp_find_unused_parameters_true",

    unfrozen_config = "blk_-1,blk_-2",
    
    img_dir = "YOUR_DIRECTORY/images/",
    gt_dir = "YOUR_DIRECTORY/siglip2_embeddings/",
)


