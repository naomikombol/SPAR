import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import AdamW

import pickle
import os

class LightningSegmentor(pl.LightningModule):
    def __init__(self, segmentor, learning_rate = 2e-5, weight_decay = 1e-4, unfrozen_config = "blk_-1, blk_-2", gt_dir = None):
        super().__init__()
        self.segmentor = segmentor

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.unfrozen_config = unfrozen_config

        self.gt_dir = gt_dir 
        if gt_dir is not None:
            os.makedirs(self.gt_dir, exist_ok=True)

        self.loss_fn = nn.MSELoss()
        if gt_dir != None:
            self.segmentor.eval()
            for param in self.segmentor.parameters():
                param.requires_grad = False
        else:
            self.segmentor.freeze_layers(unfrozen_config)



    def training_step(self, batch, batch_idx):
        img_tensors, gt_embeddings, img_name = batch

        embeddings = self.segmentor.forward_feature(img_tensors)

        loss = self.loss_fn(embeddings, gt_embeddings)
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        return 0 

    def test_step(self, batch, batch_idx):
        return 0

    def predict_step(self, batch):
        with torch.no_grad():
            img_tensors, gt_embeddings, img_name = batch
            emb_name = self.gt_dir + f"{img_name.split('/')[-1].split('.')[0]}.pkl"
            
            # If the embedding exists, do not regenerate (allows restarting)
            if os.path.isfile(emb_name):
                return None

            gt_embeddings = self.segmentor.forward_slide(img_tensors)

            with open(emb_name, "wb") as f:
                with torch.no_grad():
                    feature = gt_embeddings 
                    feature_np = feature.cpu().numpy()
                    pickle.dump(feature_np, f)
        return None

    def configure_optimizers(self):
        optimiser = AdamW([p for p in self.segmentor.parameters() if  p.requires_grad],lr=self.learning_rate,weight_decay=self.weight_decay)
        return optimiser