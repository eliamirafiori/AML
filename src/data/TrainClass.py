import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class TrainDataset(Dataset):

    def __init__(self, images_root:str, train_csv_root:str, img_size:int=256):

        df = pd.read_csv(os.path.join(train_csv_root))

        self.root = images_root
        self.image_paths = [os.path.join(images_root, f) for f in sorted(os.listdir(images_root),key=lambda x: int(x.split('_')[1].split('.')[0]))]
        self.labels = df['label'].values

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomInvert(p=0.2),
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # normalize between [0, 1]
        ])

    def __len__(self):

        return len(self.image_paths)
    
    def __getitem__(self, idx:int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        Im = Image.open(image_path).convert('RGB')
        Im = self.T(Im)

        return Im, label





