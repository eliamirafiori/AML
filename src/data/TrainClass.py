import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class TrainDataset(Dataset):

    def __init__(self, train_csv_path, imgs_dir, transform = None):



        self.df = pd.read_csv(train_csv_path)
        self.imgs_dir = imgs_dir
        self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):

            img_name   = self.df.iloc[idx]["filename"]
            label_name = self.df.iloc[idx]["label"]
            
            img_path = os.path.join(self.imgs_dir, img_name)
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            return img, label_name