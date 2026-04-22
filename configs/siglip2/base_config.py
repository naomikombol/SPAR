# base configurations
model_tokenizer_name = "hf-hub:timm/ViT-B-16-SigLIP2-512" 
model = dict(
    type='VitForSegmentation',
    wrapper = 'SigLIP2Wrapper',
    model = model_tokenizer_name, 
    tokenizer = model_tokenizer_name,
    slide_crop = int(model_tokenizer_name[-3:]),
    slide_stride =  0, 
    processing_params = dict(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5],
    )
)

