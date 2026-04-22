import random
import math
import cv2
import torch
from mmcv.transforms import BaseTransform, TRANSFORMS
from mmseg.datasets.transforms import RandomCrop

@TRANSFORMS.register_module()
class RandomCropOrFullImage:
    def __init__(self, prob_crop=0.5):
        self.prob_crop = prob_crop

    def __call__(self, results):
         
        img = results["img"]
        h, w = img.shape[0], img.shape[1]
        short_side = min(h, w)
        if short_side < 512:
            scale = 512 / short_side
            new_w = max(int(w * scale), 512)
            new_h = max( int(h * scale), 512)

                
            new_w = int(16 * math.floor(new_w / 16.))
            new_h = int(16 * math.floor(new_h / 16.))

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            results["img"] = img
            results["img_shape"] = results["img"].shape
            return results

        if random.random() > self.prob_crop:
            h, w = results["img"].shape[0], results["img"].shape[1]
            if h//16 != 0 or w//16 != 0:
                new_w = int(16 * math.floor(w / 16.))
                new_h = int(16 * math.floor(h / 16.))
                results["img"] = cv2.resize(results["img"], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                results["img_shape"] = results["img"].shape
            return results 
        
        crop_h = max(random.randint(math.floor(img.shape[0]*0.4), img.shape[0]), 512)
        crop_w = max(random.randint(math.floor(img.shape[1]*0.4), img.shape[1]), 512)

        crop_h = int(16 * math.floor(crop_h / 16.))
        crop_w = int(16 * math.floor(crop_w / 16.))

        crop_size = (crop_h, crop_w)
        crop_op =  RandomCrop(crop_size=crop_size)
        temp = crop_op(results)
        temp["img_shape"] = temp["img"].shape
        return temp


@TRANSFORMS.register_module()
class DivisibleBy16:
    def __init__(self, prob_crop=0.5):
        self.prob_crop = prob_crop

    def __call__(self, results):
         
        
        img = results["img"]
        h, w = img.shape[0], img.shape[1]
        short_side = min(h, w)
        if short_side < 512:
            scale = 512 / short_side
            new_w = max(int(w * scale), 512)
            new_h = max( int(h * scale), 512)

                
            new_w = int(16 * math.floor(new_w / 16.))
            new_h = int(16 * math.floor(new_h / 16.))

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            results["img"] = img

        h, w = results["img"].shape[0], results["img"].shape[1]
        if h//16 != 0 or w//16 != 0:
            new_w = int(16 * math.floor(w / 16.))
            new_h = int(16 * math.floor(h / 16.))
            results["img"] = cv2.resize(results["img"], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        
        results["img_shape"] = results["img"].shape[:2]

        
        return results



@TRANSFORMS.register_module()
class RandomHorizontalFlipTensor(BaseTransform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def transform(self, results):
        if random.random() < self.prob:
            img = results['img']

            if isinstance(img, torch.Tensor):
                results['img'] = torch.flip(img, dims=[2])  # Flip width-wise
        return results
