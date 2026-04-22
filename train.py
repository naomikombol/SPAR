import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from mmengine.config import Config
from pytorch_lightning import Trainer

from datasets.dataset_sa1b import SA1BDataloader
from models import SigLIP2Wrapper, OpenCLIPWrapper, DINOv3txtWrapper
from segmentors import VitForSegmentation
from lightning_segmentor import LightningSegmentor

import os
import random, numpy as np, time
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

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViT training")
    parser.add_argument("-model", type=str, default="siglip2", help="Trained model")
    args = parser.parse_args()

    seed = int(time.time()) % (2**32 - 1)
    print(f"Using random seed: {seed}")
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg_path = f"./configs/{args.model}/cfg_train_{args.model}.py"
    cfg = Config.fromfile(cfg_path)

    dm = SA1BDataloader(
        batch_size=cfg.training_params.batch_size,
        num_workers=cfg.training_params.num_workers,
        mean=cfg.processing_params.mean,
        std=cfg.processing_params.std,
        img_dir=cfg.training_params.img_dir,
        gt_dir=cfg.training_params.gt_dir,
    )

    model_wrapper = MODEL_WRAPPERS[cfg.model.wrapper](**cfg.model)
    segmentor_trained = SEGMENTORS[cfg.model.type](
        net=model_wrapper,
        **cfg.model,
    )

    model = LightningSegmentor(
        segmentor=segmentor_trained,
        learning_rate=cfg.training_params.learning_rate,
        weight_decay=cfg.training_params.weight_decay,
        unfrozen_config=cfg.training_params.unfrozen_config,
    )

    save_name = f"{cfg.model.slide_crop}_{args.model}_{cfg.training_params.unfrozen_config}"

    logs_dir = "./logs_dir"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(logs_dir, f"{save_name}_{timestamp}")

    if is_main_process():
        print("Saving config to:", run_dir)
        os.makedirs(run_dir, exist_ok=True)
        cfg.dump(os.path.join(run_dir, "config.py"))

    filename = "epoch-{epoch:02d}-train_loss-{train_loss:.4f}"
    checkpoint_dir = f"./trained_checkpoints/{save_name}"

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        verbose=True,
        dirpath=checkpoint_dir,
        filename=filename,
    )

    trainer = pl.Trainer(
        logger=False,
        strategy=cfg.training_params.strategy,
        accelerator=cfg.training_params.accelerator,
        devices=cfg.training_params.train_devices,
        min_epochs=1,
        max_epochs=cfg.training_params.train_num_epochs,
        precision=cfg.training_params.precision,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=500)],
        limit_val_batches=0,
        deterministic=False,
    )

    trainer.fit(
        model,
        datamodule=dm,
    )

    if is_main_process():
        best_path = checkpoint_callback.best_model_path
        print(f"CHECKPOINT:{best_path}")