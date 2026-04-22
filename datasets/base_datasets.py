import torch
from glob import glob
import os
from torch.utils.data import Dataset

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

import os
from PIL import Image
import pickle

@DATASETS.register_module()
class ImageOnlyDataset(BaseSegDataset):
    METAINFO = dict(classes=[], palette=[])

    def __init__(self, data_root, pipeline, img_subdir='images', img_suffix='.jpg', **kwargs):
        self.img_suffix = img_suffix.lower()
        data_prefix = dict(img_path=img_subdir)  
        super().__init__(data_root=data_root, pipeline=pipeline, data_prefix=data_prefix, **kwargs)

    def load_data_list(self):
        img_dir = os.path.join(self.data_root, self.data_prefix['img_path'])
        print(f"Looking in {img_dir}")  

        supported_exts = ('.jpg', '.png')
        data_list = []

        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith(supported_exts):
                full_path = os.path.join(img_dir, fname)
                data_list.append(dict(
                    img_path=full_path,
                    seg_map_path=None,
                    full_img_path=full_path,
                    ori_filename=fname
                ))

        if len(data_list) == 0:
            raise RuntimeError(f"No images found in {img_dir} with extensions {supported_exts}")

        return data_list

    def get_gt_seg_maps(self):
        return [None] * len(self.data_list)



@DATASETS.register_module()
class ImageAndGTEmbeddingDataset(Dataset):
    def __init__(self, img_dir = None, gt_dir = None, transform = None):
        self.img_dir = img_dir
        self.img_paths = sorted(glob(self.img_dir + "/*.png")) 
        self.gt_dir = gt_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        if self.gt_dir:
            with open(os.path.join(self.gt_dir, img_name + ".pkl"), "rb") as f:
                gt_emb = torch.tensor(pickle.load(f))
        else:
            gt_emb = torch.empty_like(img_tensor)

        return img_tensor, gt_emb, img_name
