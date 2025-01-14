import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from transforms import Preprocess
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ForestDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        # Define a default transform if none is provided
        # TODO: Use transforms suitable for the model
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Adjust as needed for RGB channels
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: Load an image from path here
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            # Convert to numpy array
            image = np.array(img)
            image = image[:,:,1:] # Removing "near-infered" channel

        # Convert numpy array to a PIL Image if needed
        #if isinstance(image, np.ndarray):
        #    image = Image.fromarray(image)

        # Convert image to RGB if it has an alpha channel
        #if image.mode == "RGBA":
            #image = image.convert("RGB") # We found out that PIL conversion to RGB keeps the "near-infered" channel which was not desired

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


class ForestDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ForestDataset(
            image_paths=self.train_data["paths"], labels=self.train_data["labels"], transform=Preprocess()
        )
        self.val_dataset = ForestDataset(
            image_paths=self.val_data["paths"], labels=self.val_data["labels"], transform=Preprocess()
        )
        self.test_dataset = ForestDataset(
            image_paths=self.test_data["paths"], labels=self.test_data["labels"], transform=Preprocess()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)