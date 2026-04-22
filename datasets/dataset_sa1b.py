import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl

from .base_datasets import ImageAndGTEmbeddingDataset


def lazy_collate(batch): 
    image_tensors, gt_emb, img_name = batch[0]
    return image_tensors.unsqueeze(0), gt_emb, img_name


class SA1BDataloader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, mean, std, img_dir, gt_dir):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.ToTensor(),                      
            T.Normalize(mean=mean,std=std),
        ])
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        

    def setup(self, stage):
        self.combined_train_dataset = ImageAndGTEmbeddingDataset(img_dir=self.img_dir,
                                                          gt_dir = self.gt_dir, 
                                                          transform= self.transform)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.combined_train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn = lazy_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.combined_train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn = lazy_collate
        )


    def test_dataloader(self):
        return DataLoader(
            self.combined_train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn = lazy_collate
        )

    def predict_dataloader(self):
        return self.val_dataloader()