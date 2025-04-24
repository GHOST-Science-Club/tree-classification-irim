from torch import nn
import torch
import numpy as np
import kornia.augmentation as kaug
from kornia import image_to_tensor


class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x):
        x_tmp = np.array(x)
        x_out = image_to_tensor(x_tmp, keepdim=True)
        return x_out.float() / 255.0


class LambdaTransform(nn.Module):
    """Custom Lambda transform to replace Kornia's Lambda."""
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Transforms(nn.Module):
    def __init__(self, image_size=(224, 224)):
        """
        Class handles transformation of images
        """
        super().__init__()

        self.train_transforms = nn.Sequential(
            LambdaTransform(lambda x: image_to_tensor(x, keepdim=True).float() / 255.0),
            kaug.Resize(size=image_size),
            kaug.RandomRotation(degrees=360.0, p=1.0),
            kaug.RandomHorizontalFlip(p=0.5),
            kaug.RandomVerticalFlip(p=0.5),
            kaug.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.7, 1.30), p=0.5),
            kaug.RandomChannelShuffle(p=0.05),
            kaug.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
            # kaug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0), # This tranformation is making some issues while images are not in RGB or grayscale
            # kaug.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, p=1.0),
            # kaug.RandomPerspective(distortion_scale=0.2, p=1.0),
            # kaug.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
        )

        self.test_transforms = nn.Sequential(
            LambdaTransform(lambda x: image_to_tensor(x, keepdim=True).float() / 255.0),
            kaug.Resize(size=image_size),
            kaug.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        )

    def forward(self, x, train=True):
        if train:
            return self.train_transforms(x).squeeze(0)
        else:
            return self.test_transforms(x).squeeze(0)
