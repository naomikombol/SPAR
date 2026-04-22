import sys
sys.path.insert(0, "/home/nkombol/karantena")

from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS
init_default_scope('mmseg')

from tqdm import tqdm
import argparse
import os
import torchvision.transforms as transforms

from utils.image_generation_utils import RandomCropOrFullImage, RandomHorizontalFlipTensor, DivisibleBy16
from datasets.base_datasets import ImageOnlyDataset



parser = argparse.ArgumentParser(description="Run ViT embedding generation")
parser.add_argument("-config", type=str, help="Image generation config")
args = parser.parse_args()

cfg_path = f"./configs/image_generation/{args.config}.py"
cfg = Config.fromfile(cfg_path)

if cfg.img_dir:
    os.makedirs(cfg.img_dir, exist_ok=True)

dataset = DATASETS.build(cfg.generation_dataloader.dataset)
cfg.generation_dataloader.dataset = dataset
runner = Runner.build_dataloader(cfg.generation_dataloader)

for img in tqdm(runner):
    img_path = img["img_path"][0]

    for image in img["img"]:
        image_to_process = transforms.functional.to_pil_image(image[[2, 1, 0], :, :])
        
        image_to_process.save(f"{cfg.img_dir}/{img_path.split('/')[-1].split('.')[0]}.png")

 
    