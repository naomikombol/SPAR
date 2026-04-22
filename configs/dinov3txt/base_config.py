# base configurations
model = dict(
    type='VitForSegmentation',
    wrapper = 'DINOv3txtWrapper',
    model = 'dinov3_vitl16_dinotxt_tet1280d20h24l',
    weights = "YOUR_DIRECTORY/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    backbone_weights = "YOUR_DIRECTORY/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    git_name = "facebookresearch/dinov3",
    slide_crop = 384,
    slide_stride =  0, 
    processing_params = dict(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
)




