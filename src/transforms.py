from torch import nn
import torch
import numpy as np
import kornia.augmentation as kaug
from kornia import image_to_tensor
from random import choice, random


class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x):
        x_tmp = np.array(x)
        x_out = image_to_tensor(x_tmp, keepdim=True)
        return x_out.float() / 255.0


class Transforms(nn.Module):
    def __init__(self, prob: float = 0.5, test: bool = False):
        """
        Class handles transformation of images:
        Params:
          prob (floating) - probability of applying transformation
          test (boolean) - specify if the current transformation works on train or test set
        """
        super().__init__()
        self.prob = prob
        self.test = test

        self.train_transforms = [
            kaug.RandomHorizontalFlip(p=1.0),
            kaug.RandomVerticalFlip(p=1.0),
            kaug.RandomRotation(degrees=30, p=1.0),
            # kaug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0), # This tranformation is making some issues while images are not in RGB or grayscale
            kaug.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, p=1.0),
            kaug.RandomPerspective(distortion_scale=0.2, p=1.0),
            kaug.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
        ]

        self.resize = kaug.Resize(size=(224, 224))

    @torch.no_grad()
    def forward(self, x):

        if self.test or random() < self.prob:
            result = self.resize(x)
            return result.squeeze()

        else:
            transform = choice(
                self.train_transforms)  # In paper method applies only one transformation at once, might change in the future
            result = transform(self.resize(x))
            return result.squeeze()
