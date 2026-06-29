import argparse
from io import BytesIO
import tempfile

import requests
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mmengine.config import Config
from mmengine.registry import init_default_scope

from models import SigLIP2Wrapper, OpenCLIPWrapper, DINOv3txtWrapper
from segmentors import VitForSegmentation

init_default_scope("mmseg")

MODEL_WRAPPERS = {
    "OpenCLIPWrapper": OpenCLIPWrapper,
    "SigLIP2Wrapper": SigLIP2Wrapper,
    "DINOv3txtWrapper": DINOv3txtWrapper,
}

SEGMENTORS = {
    "VitForSegmentation": VitForSegmentation,
}

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", type=int, default=0)
parser.add_argument("-model", type=str, required=True)
parser.add_argument("-checkpoint", type=str, required=True)
args = parser.parse_args()

device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
)

cfg_path = f"./configs/{args.model}/base_config.py"
cfg = Config.fromfile(cfg_path)


image = Image.open("/home/nkombol/SPAR/frankfurt_000000_000294_leftImg8bit.png").convert("RGB")

image = torchvision.transforms.functional.to_tensor(image)*255  # RGB, CHW, float32
image = image[[2, 1, 0]]   # RGB → BGR - quirk of processing pipeline, expects images in BGR

classes = ["background", "building", "car", "road", "sidewalk", "tree", "sky"]

with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
    f.write("\n".join(classes))
    f.flush()

    cfg.model.name_path = f.name

    model_wrapper = MODEL_WRAPPERS[cfg.model.wrapper](device=device, **cfg.model).to(device)
    segmentor = SEGMENTORS[cfg.model.type](net=model_wrapper, device=device, **cfg.model)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if args.checkpoint.endswith("ckpt"):
            checkpoint = checkpoint["state_dict"]
            checkpoint = {k[len("segmentor.net."):]: v for k, v in checkpoint.items() if k.startswith("segmentor.net.") }
        segmentor.net.load_state_dict(checkpoint, strict = True)

    segmentor.eval()

    sample = {
    "inputs": [image]
    }

    with torch.no_grad():
        processed = segmentor.data_preprocessor(sample,training=False)
        processed["inputs"] = processed["inputs"].to(device)
        result = segmentor.predict(processed)
        pred_mask = result[0]["pred_sem_seg"]["data"].cpu().numpy()

        # Image-saving and legend logic - pred_mask may be saved as is, this is for nicer rendering and interpretation
        np.random.seed(42)
        H, W = pred_mask.shape
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
        colored = colors[pred_mask]
        present = np.unique(pred_mask)

        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1]) 

        original = image[[2, 1, 0]].byte().permute(1, 2, 0).cpu().numpy()
        ax.imshow(original)
        ax.imshow(colored, interpolation="nearest", alpha=0.5)
        ax.axis("off")

        handles = [
            mpatches.Patch(
                color=colors[idx] / 255.0,
                label=str(classes[idx])
            )
            for idx in present
        ]

        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            fontsize=20
        )

        plt.savefig("predictions.png", dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close()