import torch
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch >= 2.31.0, timm >= 1.0.15
import math
import torch.nn.functional as F

class SigLIP2Wrapper(torch.nn.Module):
    def __init__(self,**model_dict):
        super().__init__()
        self.model_name = model_dict["model"]
        self.model, _ = create_model_from_pretrained(self.model_name)
        self.model.eval()

        self.tokenizer = get_tokenizer(model_dict["tokenizer"])

        self.dtype = self.model.visual.trunk.patch_embed.proj.weight.dtype
        self.patch_size = self.model.visual.trunk.patch_embed.proj.kernel_size[0]


    def encode_image(self, image):
        return self.forward(image)

    def encode_text(self, text):
        return self.model.encode_text(text)


    def forward(self, x: torch.Tensor):
        img_shape = x.shape

        self.model.visual.trunk.patch_embed.strict_img_size =  False
        self.model.visual.trunk.patch_embed.dynamic_img_pad = True

        x = self.model.visual.trunk.patch_embed(x)

        
        # Dynamic positional embedding resizing
        patched_size = (-(img_shape[-2] // -self.patch_size)  , -(img_shape[-1] // -self.patch_size))

        pos_embed = self.model.visual.trunk.pos_embed 
        pos_w_h = math.isqrt(pos_embed.shape[-2])

        if img_shape[-2:] != (pos_w_h, pos_w_h):
            pos_embed = pos_embed.reshape(1, pos_w_h, pos_w_h, pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(-(img_shape[-2] // -16)  , -(img_shape[-1] // -16))
            , mode='bicubic').reshape(1,pos_embed.shape[1], -1 )
            pos_embed = pos_embed.permute(0, 2, 1)

        x = x + pos_embed
        x = self.model.visual.trunk.pos_drop(x)
        x = self.model.visual.trunk.patch_drop(x)
        x = self.model.visual.trunk.norm_pre(x)
        
        
        x = self.model.visual.trunk.blocks(x)
        x = self.model.visual.trunk.norm(x)

        attn_layer = self.model.visual.trunk.attn_pool
        B, N, C = x.shape 
        
        # Skipping SigLIP2's attention pool, patch embeddings go straight to value space and then the out_projection
        kv = attn_layer.kv(x).reshape(B, N, 2, attn_layer.num_heads, attn_layer.head_dim).permute(2, 0, 3, 1, 4)
        _, v = kv.unbind(0)
        v = v.transpose(1, 2).reshape(B, N, C)
        v = attn_layer.proj(v)
        v = attn_layer.proj_drop(v)

        x = v + attn_layer.mlp(attn_layer.norm(v)) 

        feat_shape = x.shape

        x = x.reshape(feat_shape[0], patched_size[0],patched_size[1], feat_shape[-1]) 

        return x
    
    def freeze_layers(self, unfrozen_config):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        blocks = self.model.visual.trunk.blocks
        
        for unfrozen in unfrozen_config.split(","):
            unfrozen = unfrozen.strip()
            parts = unfrozen.split("_")
            prefix = parts[0]
            idx = int(parts[-1]) if len(parts) > 1 else None

            if prefix in {"blk", "qkv", "mlp"}:
                block = blocks[idx]

                for name, param in block.named_parameters():
                    if (
                        prefix == "blk"
                        or (prefix == "qkv" and "qkv" in name)
                        or (prefix == "mlp" and "mlp" in name)
                    ):
                        param.requires_grad = True
            elif prefix == "attn_pool_kv":
                for name, param in self.model.named_parameters():
                    if "visual" in name and "attn_pool" in name and "kv" in name:
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


    
    


