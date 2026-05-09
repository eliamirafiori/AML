import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os


class CLIPCSVDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        # Get unique classes for the Discriminative Task (Eq. 2)
        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["filename"]
        label_name = self.df.iloc[idx]["label"]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[label_name]
