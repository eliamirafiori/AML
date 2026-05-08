import models.sam as sam
import torch
import torchvision.transforms as T
import random
import numpy as np



def augmentation_with_sam(img,sam_model):

    h, w, _ = img.shape
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1]) # suggerimento positivo: l'oggetto si trova sotto queste coordinate

    masks = sam_model.predict(img, point_coords=input_point, point_labels=input_label)
    mask = masks[0]

    # Esempio di augmentation (soggetto + il resto sfondo nero)
    augmented_image = img.copy()
    augmented_image[~mask] = 0 # metto sfondo nero dove non c'è il soggetto

    return augmented_image


def random_augmentation(img):
    # Esempio di augmentation casuale: rotazione, flip, scaling
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.RandomResizedCrop((img.shape[0], img.shape[1]), scale=(0.8, 1.0)),
        T.ToTensor()
    ])
    augmented_image = transform(img)
    return augmented_image.permute(1, 2, 0).numpy()
