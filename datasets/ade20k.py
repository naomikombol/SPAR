import os
import glob
import PIL.Image
from torch.utils import data

class ADE20K(data.Dataset):
    def __init__(self, images_path, target_path,  transform=None):
        super(ADE20K, self).__init__()
        self.img_files = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
        self.target_files = sorted(glob.glob(os.path.join(target_path, '*.png')))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index]
        target_path = self.target_files[index]

        img = PIL.Image.open(img_path).convert('RGB')
        target = PIL.Image.open(target_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.img_files)
