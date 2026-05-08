import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class TrainingDataClass(Dataset):

    def __init__(self, train_csv, imgs_dir, transform=None):

        self.df = pd.read_csv(train_csv)
        self.imgs_dir = imgs_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img_name = self.df.iloc[idx, 0] # Colonna 'filename'
        label = self.df.iloc[idx, 1]    # Colonna 'label'

        img_path = os.path.join(self.imgs_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)




















    
   

    

    
        
    
        

    