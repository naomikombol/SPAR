import sys 
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

def unfold_with_boundary(img, window_size, stride):
    B, C, H, W = img.shape
    h_win, w_win = window_size
    stride_h, stride_w = stride
    
    # Calculate all valid starting positions
    h_starts = list(range(0, H - h_win + 1, stride_h))
    w_starts = list(range(0, W - w_win + 1, stride_w))
    
    # Ensure last boundary-window is included
    if h_starts[-1] != H - h_win:
        h_starts.append(H - h_win)
    if w_starts[-1] != W - w_win:
        w_starts.append(W - w_win)
    
    patches = []
    positions = []
    for b in range(B):
        for hs in h_starts:
            for ws in w_starts:
                patch = img[b:b+1, :, hs:hs+h_win, ws:ws+w_win]
                patches.append(patch)
                positions.append((b, hs, ws))
    
    patches = torch.cat(patches, dim=0)  
    return patches, positions



def fold_from_patches(patches, positions, recon, count, reduction_factor):
    h, w = patches.shape[-2:]
    positions = torch.tensor(positions) 
    for i in range(patches.size(0)):
        b = positions[i, 0]
        hs = positions[i, 1] // reduction_factor
        ws = positions[i, 2] // reduction_factor

        recon[b:b+1, :, hs:hs+h, ws:ws+w].add_(patches[i:i+1])
        count[b:b+1, :, hs:hs+h, ws:ws+w].add_(1)

    return recon, count

def finalize_fold(recon, count):
    recon.div_(count.clamp(min=1))
    return recon


class VitForSegmentation(BaseSegmentor):
    def __init__(self, net = None, **model_dict): 
        if "processing_params" in model_dict:
            data_preprocessor = SegDataPreProcessor(
                mean = [val*255 for val in model_dict["processing_params"]["mean"]],
                std= [val*255 for val in model_dict["processing_params"]["std"]],
                bgr_to_rgb=True
            )
        else:
            data_preprocessor = SegDataPreProcessor(
                mean=[0.5*255, 0.5*255, 0.5*255],
                std=[0.5*255, 0.5*255, 0.5*255],
                bgr_to_rgb=True
            )
        super().__init__(data_preprocessor=data_preprocessor)

        self.net = net
        self.tokenizer = self.net.tokenizer
        
        self.windowslide_bs = model_dict.get("gen_windowslide_bs", 10)
        if "name_path" in model_dict:
            self.windowslide_bs = model_dict.get("eval_windowslide_bs", 60)
            self.name_path = model_dict["name_path"]

            query_words, self.query_idx = get_cls_idx(self.name_path)
            self.num_queries = len(query_words)
            self.num_classes = max(self.query_idx) + 1
            self.query_idx = torch.Tensor(self.query_idx).to(torch.int64)

            query_features = []
            with torch.no_grad():
                device = next(self.net.model.parameters()).device
                for qw in query_words:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)

                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))

            self.query_features = torch.cat(query_features, dim=0)
        
        self.dtype = self.net.dtype

        self.logit_scale = model_dict.get("logit_scale",40)
        self.logit_bias = model_dict.get("logit_bias",0)

        self.prob_thd = model_dict.get("prob_thd",0.0)
        self.area_thd = model_dict.get("area_thd",None)
        self.slide_stride = model_dict.get("slide_stride", 24)
        self.slide_crop = model_dict.get("slide_crop",512)

        def compute_upsample_factor(patch_size, stride):
            for r in range(1, patch_size + 1):
                if patch_size % r == 0 and stride % (patch_size // r) == 0:
                    return r
                
        if self.slide_stride != 0:
            self.upsample_factor =  compute_upsample_factor(self.net.patch_size, self.slide_stride)
            if "name_path" not in model_dict:
                self.reduction_factor = self.net.patch_size // self.upsample_factor
            else:
                self.reduction_factor = 1
        else:
            self.upsample_factor, self.reduction_factor = 0, 0
       
        self.align_corners = False


    def freeze_layers(self, unfrozen_config):
        self.net.freeze_layers(unfrozen_config)

    def forward_feature(self, img, original_size = None):
        if type(img) == list:
            img = img[0]

        image_features = self.net.encode_image(img)      

        if self.upsample_factor != 0 and not hasattr(self, "name_path"):
            image_features = image_features.permute(0, 3, 1, 2) 
            image_features = nn.functional.interpolate(image_features, size=(int(image_features.shape[-2] * self.upsample_factor),
                                                                            int(image_features.shape[-1] * self.upsample_factor)), mode='bilinear')
        
        if hasattr(self, "name_path"):
            image_features =  image_features/ image_features.norm(dim=-1, keepdim=True)   
            logits = image_features @ self.query_features.T

            logits = logits.permute(0, 3, 1, 2)
            logits = nn.functional.interpolate(logits, size=original_size, mode='bilinear')
            return logits

        return image_features
        
    def forward_slide(self, img,  original_size = None):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(self.slide_stride) == int:
            slide_stride = (self.slide_stride, self.slide_stride)
        if type(self.slide_crop) == int:
            self.crop_size = (self.slide_crop, self.slide_crop)

        temp, positions = unfold_with_boundary(img, self.crop_size, slide_stride)
        
        num_samples = temp.shape[0]

        B = img.size(0)          

        if hasattr(self, "name_path"):
            C = self.query_features.shape[0] 
        else:
            if "clip" in str(type(self.net)).split(".")[-1].lower():
                C = 512 
            elif "siglip" in str(type(self.net)).split(".")[-1].lower():
                C = 768
            elif "dino" in str(type(self.net)).split(".")[-1].lower():
                C = 1024
                                        

        H, W = img.size(-2), img.size(-1)
        device = temp.device

        recon = torch.zeros(B, C, H // self.reduction_factor, W // self.reduction_factor, device=device)
        count = torch.zeros(B, 1, H // self.reduction_factor, W // self.reduction_factor, device=device)


        for i in range(0, num_samples, self.windowslide_bs):
            batch = temp[i:i+self.windowslide_bs]
            pos_batch = positions[i:i+self.windowslide_bs]

            embeddings = self.forward_feature(batch)
            recon, count = fold_from_patches(embeddings, pos_batch, recon, count, reduction_factor = self.reduction_factor)
            
        embeddings = finalize_fold(recon, count)     

        if not hasattr(self, "name_path"): 
            embeddings = nn.functional.interpolate(embeddings, size=(img.size(-2) // self.net.patch_size, img.size(-1)// self.net.patch_size), mode='bilinear')
            return embeddings.permute(0, 2, 3, 1) 
        elif original_size != img.size[-2:]:
            logits = nn.functional.interpolate(logits, size=original_size, mode='bilinear')
        return embeddings

    def predict(self, data):
        if data["data_samples"] is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data["data_samples"]
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=data["inputs"].shape[2:],
                    img_shape=data["inputs"].shape[2:],
                    pad_shape=data["inputs"].shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * data["inputs"].shape[0]
        
        if self.slide_stride > 0:
            seg_logits = self.forward_slide(data["inputs"], batch_img_metas[0]["ori_shape"]) 
        else:
            seg_logits = self.forward_feature(data["inputs"], batch_img_metas[0]["ori_shape"])

        return self.postprocess_result(seg_logits, data["data_samples"])
    
    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale + self.logit_bias
            seg_logits = seg_logits.softmax(0) # n_queries * w * h
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx).to(self.query_features)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(torch.float)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype) 
                seg_logits[1:] *= area_pred.transpose(0, -1)

            seg_pred = seg_logits.argmax(0, keepdim=True)
            
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices