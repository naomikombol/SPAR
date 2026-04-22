import torch
import torch.nn.functional as F
import open_clip
import math

class OpenCLIPWrapper(torch.nn.Module):
    def __init__(self, **model_dict):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_dict["clip_path"], 
            pretrained=model_dict["pretrained"],  quick_gelu="openai" in model_dict["pretrained"])
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_dict["clip_path"])
        self.model_name = model_dict["clip_path"]
        self.pretrained = model_dict["pretrained"]

        self.dtype = self.model.visual.conv1.weight.dtype
        self.patch_size = self.model.visual.conv1.kernel_size[0]

    def encode_image(self, image):
        return self.forward(image)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.model.visual.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.model.visual.positional_embedding
        class_pos_embed = self.model.visual.positional_embedding[[0]]
        patch_pos_embed = self.model.visual.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic', antialias = False,
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor):
        img_shape = x.shape
        patched_size = img_shape[-2] // 16  , img_shape[-1] // 16

        B, nc, w, h = x.shape
        x = self.model.visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  

        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if x.shape[1] != self.model.visual.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)       
        
        if not self.model.visual.transformer.resblocks[0].attn.batch_first:  # NLD -> LND if needed
                x = x.permute(1, 0, 2)  
        
        for blk in self.model.visual.transformer.resblocks[:-1]:
            x = blk(x)

        for blk in self.model.visual.transformer.resblocks[-1:]:
            if self.model.visual.transformer.resblocks[0].attn.batch_first: 
                x = x.permute(1, 0, 2)  # NLD -> LND 
            
            x = x + self.custom_attn_maskclip(blk.attn, blk.ln_1(x))
            x = x + blk.mlp(blk.ln_2(x))
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.visual.ln_post(x) @ self.model.visual.proj
        x = x[:, 1:]
        
        feat_shape = x.shape
        x = x.reshape(feat_shape[0], patched_size[0],patched_size[1], feat_shape[-1]) 

        return x

    
    def custom_attn_maskclip(self, attn_layer, x):
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads

        _, _, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        v = v.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        v = attn_layer.out_proj(v)

        return v

    def freeze_layers(self, unfrozen_config):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        blocks = self.model.visual.transformer.resblocks

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

    
