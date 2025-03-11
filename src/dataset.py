import os
import yaml
import psutil
import numpy as np
from PIL import Image
from math import floor
import pytorch_lightning as pl
from transforms import Preprocess
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


with open("src/config.yaml", "r") as c:
    config = yaml.safe_load(c)


def calculate_dataloader_params(batch_size, img_size=(224, 224), image_channels=3, precision=32, ram_fraction=0.8):
    """
    Function calculates the number of workers and prefetch factor 
    for DataLoader based on the available RAM.

    Input:
        batch_size: int - the batch size used in DataLoader
        img_size: int - the size of the image
        image_channels: int - the number of channels in the image
        precision: int - the precision of the weights
        ram_fraction: float - the fraction of RAM to use
    Output:
        dict of params: num_workers, prefetch_factor, pin_memory, persistent_workers
            num_workers: int - the number of workers
            prefetch_factor: int - the prefetch factor
            pin_memory: bool - whether to use pin_memory
            persistent_workers: bool - whether to use persistent workers
    """
    if config['training']['dataloader']['auto']:
        total_ram = psutil.virtual_memory().available * ram_fraction
        img_memory = np.prod(img_size) * image_channels * (precision/8)
        batch_memory = batch_size * img_memory

        if batch_memory > total_ram:
            raise ValueError("Batch size too large for available RAM. Reduce the batch size or image dimensions.")

        max_batches_in_ram = floor(total_ram / batch_memory)
        
        prefetch_factor = min(max_batches_in_ram, 16)
        num_workers = min(floor(prefetch_factor / 2), os.cpu_count())

        params = {"num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "pin_memory": config['device'] == 'gpu',
                "persistent_workers": True}
        
    else:
        params = {"num_workers": config['training']['dataloader']['num_workers'],
                  "prefetch_factor": config['training']['dataloader']['prefetch_factor'],
                  "pin_memory": config['training']['dataloader']['pin_memory'],
                  "persistent_workers": config['training']['dataloader']['persistent_workers']}


    return params

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
            image = image[:, :, 1:] if image.shape[-1] == 4 else image  # Removing "near-inferred" channel
        # We found out that PIL conversion to RGB
        # keeps the "near-inferred" channel which was not desired

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


class ForestDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.params = calculate_dataloader_params(batch_size)

    def setup(self, stage=None):
        self.train_dataset = ForestDataset(
            image_paths=self.train_data["paths"],
            labels=self.train_data["labels"], transform=Preprocess()
        )
        self.val_dataset = ForestDataset(
            image_paths=self.val_data["paths"],
            labels=self.val_data["labels"], transform=Preprocess()
        )
        self.test_dataset = ForestDataset(
            image_paths=self.test_data["paths"],
            labels=self.test_data["labels"], transform=Preprocess()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          **self.params)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          **self.params)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          **self.params)



if __name__ == '__main__':
    params = calculate_dataloader_params(32)
    print(params)