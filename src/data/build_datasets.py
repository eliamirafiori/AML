import os

from torch.utils.data import Subset, DataLoader
from torchvision import datasets


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

    few_shot_set = Subset(dataset, indices) # Create a subset of the dataset using the selected indices

    return DataLoader(few_shot_set, batch_size=batch_size, shuffle=True) # Create a DataLoader for the few-shot dataset with shuffling enabled
