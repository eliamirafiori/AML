import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

GLOBAL_PATH = os.getcwd()
IMGS_PATH = os.path.join(GLOBAL_PATH, 'src','data', 'release','test_images')
TRAIN_CSV_PATH = os.path.join(GLOBAL_PATH, 'src','data', 'release','test_episodes_release.csv')


class TestDataset(Dataset):

    def __init__(self, images_root:str, test_csv_root:str, episode_idx:int, img_size:int=256):

        df = pd.read_csv(os.path.join(test_csv_root), index_col=0)
        df = df[df['episode_id'] == episode_idx]

        self.root = images_root
        self.labels = [int(x) for x in df['label'].values if pd.notnull(x)]
        self.images_paths = [os.path.join(IMGS_PATH,fname) for fname in df['filename'].values]

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_size, img_size))])
        

    def __len__(self):
        return len(self.images_paths)
    

    def __getitem__(self, idx:int):
        image_path = self.images_paths[idx]
        label = self.labels[idx]
        Im = Image.open(image_path).convert('RGB')
        Im = self.T(Im)

        return Im, label



test_dataset = TestDataset(IMGS_PATH, TRAIN_CSV_PATH, 1, 256)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)






