import os

from torch.utils.data import Subset, DataLoader
from torchvision import datasets

from src.data.csv_dataset import CLIPCSVDataset


def build_train_set():
    os.makedirs("./train", exist_ok=True)

    with open("./release/train.csv", "r") as f:
        next(f)
        for line in f:
            filename, label = line.split(",")
            os.rename(f"./release/images/{filename}", f"./train/{filename}")

    os.makedirs("./test", exist_ok=True)
    for filename in os.listdir("./release/images"):
        os.rename(f"./release/images/{filename}", f"./test/{filename}")


def get_few_shot_loader(root, transform, n_shots=16, batch_size=32):

    dataset = datasets.ImageFolder(root, transform=transform) # Assumes a directory structure where each subdirectory is a class

    indices = []
    class_counts = {} # Dictionary to keep track of how many samples we've added for each class

    # Simple sampling logic: dataset.samples is a list of (filepath, class_index) tuples
    for idx, (_, label) in enumerate(dataset.samples):
        class_counts[label] = class_counts.get(label, 0)
        if class_counts[label] < n_shots:
            indices.append(idx)
            class_counts[label] += 1

    few_shot_set = Subset(dataset, indices)
    return DataLoader(few_shot_set, batch_size=batch_size, shuffle=True)


from torch.utils.data import Subset, DataLoader


def get_csv_few_shot_loader(csv_path, img_dir, transform, n_shots=5, batch_size=32):
    full_dataset = CLIPCSVDataset(csv_path, img_dir, transform)

    # Create a balanced few-shot subset
    indices = []
    counts = {cls: 0 for cls in full_dataset.classes}

    # Iterate through the dataframe and pick shots
    for i, row in full_dataset.df.iterrows():
        cls = row["label"]
        if counts[cls] < n_shots:
            indices.append(i)
            counts[cls] += 1

    few_shot_dataset = Subset(full_dataset, indices)
    return (
        DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True),
        full_dataset.classes,
    )
