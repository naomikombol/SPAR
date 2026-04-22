import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from mmengine.config import Config
from pytorch_lightning import Trainer

from models import SigLIP2Wrapper, OpenCLIPWrapper, DINOv3txtWrapper
from segmentors import VitForSegmentation
from lightning_segmentor import LightningSegmentor
from datasets.dataset_sa1b import SA1BDataloader

import argparse

torch.set_float32_matmul_precision("medium") 

MODEL_WRAPPERS = {
    "OpenCLIPWrapper": OpenCLIPWrapper,
    "SigLIP2Wrapper": SigLIP2Wrapper,
    "DINOv3txtWrapper": DINOv3txtWrapper,
}
SEGMENTORS = {
    "VitForSegmentation": VitForSegmentation,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViT embedding generation")
    parser.add_argument("-model", type=str, default="siglip2", help="Teacher model")
    args = parser.parse_args()

    cfg_path = f"./configs/{args.model}/cfg_train_{args.model}.py"
    cfg = Config.fromfile(cfg_path)
    cfg.model.slide_stride = cfg.model.slide_stride_gen

    model_wrapper = MODEL_WRAPPERS[cfg.model.wrapper](**cfg.model)
    segmentor_slide = SEGMENTORS[cfg.model.type](
        net = model_wrapper, 
        **cfg.model,
    )
    model = LightningSegmentor(
        segmentor = segmentor_slide,
        gt_dir = cfg.training_params.gt_dir
    )
    
    dm = SA1BDataloader(
        batch_size=cfg.training_params.batch_size,
        num_workers=cfg.training_params.num_workers,
        mean = cfg.processing_params.mean,
        std = cfg.processing_params.std,
        img_dir = cfg.training_params.img_dir,
        gt_dir = None
    )


    trainer = pl.Trainer(
        strategy = cfg.training_params.strategy,
        accelerator = cfg.training_params.accelerator,
        devices = cfg.training_params.gen_devices,
        max_epochs = cfg.training_params.gen_num_epochs,
        precision = cfg.training_params.precision,
        num_sanity_val_steps=0,
    )

    predictions = trainer.predict(model, datamodule=dm)
