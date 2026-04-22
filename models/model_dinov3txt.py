import torch
import torch.nn.functional as F
import math

class DINOv3txtWrapper(torch.nn.Module):
    def __init__(self, **model_dict): 
        super().__init__()
        self.model, self.tokenize_model = torch.hub.load(model_dict["git_name"], model_dict["model"], 
                                                                 weights = model_dict["weights"], 
                                                                 backbone_weights=model_dict["backbone_weights"])
        self.tokenizer = self.tokenize_model.tokenize

        self.model.eval()
        self.model_name = model_dict["model"]

        self.dtype = self.model.visual_model.backbone.patch_embed.proj.weight.dtype
        self.patch_size = self.model.visual_model.backbone.patch_embed.proj.kernel_size[0]

    
    def encode_image(self, img):
        B, _, H, W = img.shape
        P = self.model.visual_model.backbone.patch_size
        new_H = math.ceil(H / P) * P
        new_W = math.ceil(W / P) * P

        # Stretch image to a multiple of patch size
        if (H, W) != (new_H, new_W):
            img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)  

        B, _, h_i, w_i = img.shape

        _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(img)
        blocks_patches = (
            patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
        ) 

        return blocks_patches


    def encode_text(self, text):
        return  self.model.text_model(torch.tensor(text))[:, 1024:]

    def freeze_layers(self, unfrozen_config):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        blocks = self.model.visual_model.backbone.blocks

        for unfrozen in unfrozen_config.split(","):
            unfrozen = unfrozen.strip()
            parts = unfrozen.split("_")
            prefix = parts[0]
            idx = int(parts[-1]) if len(parts) > 1 else None

            if prefix in {"blk", "qkv", "mlp"}:
                block = blocks[idx]
                for name, param in block.named_parameters():
                    if prefix == "blk" \
                    or (prefix == "qkv" and "qkv" in name) \
                    or (prefix == "mlp" and "mlp" in name):
                        param.requires_grad = True
            elif prefix == "posembed":
                for name, param in self.model.named_parameters():
                    if "visual" in name and "pos_embed" in name:
                        param.requires_grad = True
            elif prefix == "patchembed":
                for name, param in self.model.named_parameters():
                    if "visual" in name and "patch_embed" in name:
                        param.requires_grad = True
            
        for name, param in self.model.named_parameters():
            if unfrozen_config == "all":
                param.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        


    