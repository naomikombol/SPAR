import argparse

import pickle
import torch
import torchvision
from PIL import Image

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.models.data_preprocessor import SegDataPreProcessor

from models import SigLIP2Wrapper, OpenCLIPWrapper, DINOv3txtWrapper

init_default_scope("mmseg")

MODEL_WRAPPERS = {
    "OpenCLIPWrapper": OpenCLIPWrapper,
    "SigLIP2Wrapper": SigLIP2Wrapper,
    "DINOv3txtWrapper": DINOv3txtWrapper,
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

data_preprocessor = SegDataPreProcessor(
    mean = [val*255 for val in cfg.model["processing_params"]["mean"]],
    std= [val*255 for val in cfg.model["processing_params"]["std"]],
    bgr_to_rgb=True
)


image = Image.open("/home/nkombol/SPAR/frankfurt_000000_000294_leftImg8bit.png").convert("RGB")

image = torchvision.transforms.functional.to_tensor(image)*255  # RGB, CHW, float32
image = image[[2, 1, 0]]   # RGB → BGR - quirk of mmseg processing pipeline, expects images in BGR


model_wrapper = MODEL_WRAPPERS[cfg.model.wrapper](device=device, **cfg.model).to(device)

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # If using checkpoints you've previously trained, they are saved for continued training so require "trimming"
    if args.checkpoint.endswith("ckpt"):
        checkpoint = checkpoint["state_dict"]
        checkpoint = {k[len("segmentor.net."):]: v for k, v in checkpoint.items() if k.startswith("segmentor.net.") }
    model_wrapper.load_state_dict(checkpoint, strict = True)



with torch.no_grad():
    # Datapreprocessor expects a dictionary, returns image in RGB; if you want to preprocess yourself, model_wrapper takes RGB normalised tensor input
    sample = {
    "inputs": [image]
    }
    processed = data_preprocessor(sample,training=False)
    
    # The only important thing is that the image is properly normalized, for siglip2 that is 0.5 mean and var
    processed["inputs"] = processed["inputs"].to(device)

    # Uninterpoalted and unnormalized embeddings, user may adjust as needed -> make sure to interpolate before normalisation!
    embeddings = model_wrapper.forward(processed["inputs"])
    
    # Generated embeddings for SPAR training are also saved as .pkl; unlike those, these are unnormalized
    with open("embeddings.pkl", "wb") as f:
        embeddings_np = embeddings.cpu().numpy()
        pickle.dump(embeddings_np, f)
        