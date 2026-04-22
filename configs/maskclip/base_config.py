# base configurations
model = dict(
    type='VitForSegmentation',
    wrapper = 'OpenCLIPWrapper',
    pretrained = 'laion2b_s34b_b88k',
    clip_path='ViT-B-16',
    slide_crop = 224,
    slide_stride =  0, 
    processing_params = dict(
        mean = [0.48145466, 0.4578275, 0.40821073],
        std = [0.26862954, 0.26130258, 0.27577711],
    )
)


