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
import random


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
        self.transform = transforms.Compose(
            [
                Preprocess(),
                transform
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


class UndersampledDataset(ForestDataset):
    def __init__(self, image_paths, labels, transform=None, target_size=None):
        super().__init__(image_paths, labels, transform)

        class_indices = {}
        for idx, label in enumerate(labels):
            class_indices.setdefault(label, []).append(idx)

        # Find the minimum number of samples in a class
        min_count = min(len(indices) for indices in class_indices.values())

        # If the target_size is not provided, set it to the minimum count
        target_size = target_size if target_size else min_count

        self.sampled_indices = []
        for indices in class_indices.values():
            # Limit the number of images per class to target_size (if it exceeds the target_size)
            self.sampled_indices.extend(random.sample(indices, min(target_size, len(indices))))

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.sampled_indices[idx])


class OversampledDataset(ForestDataset):
    def __init__(self, image_paths, labels, transform=None, minority_transform=None, oversample_factor=2,
                 oversample_threshold=200):
        super().__init__(image_paths, labels, transform)
        self.minority_transform = minority_transform

        class_indices = {}
        for idx, label in enumerate(labels):
            class_indices.setdefault(label, []).append(idx)

        self.to_transform = set()
        self.sampled_indices = []
        for label, indices in class_indices.items():
            if len(indices) <= oversample_threshold:
                self.to_transform.add(label)
                # Sampling the minority class with replacement
                self.sampled_indices.extend(random.choices(indices, k=int(oversample_factor * len(indices))))
            else:
                self.sampled_indices.extend(indices)

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        image, label = super().__getitem__(self.sampled_indices[idx])
        if label in self.to_transform and self.minority_transform:
            image = self.minority_transform(image)
        return image, label


class CurriculumLearningDataset(ForestDataset):
    def __init__(self, image_paths, labels, indices, transform=None):
        super().__init__(image_paths, labels, transform)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])


class ForestDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, dataset, dataset_args={}, batch_size=32):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.dataset = dataset
        self.dataset_args = dataset_args
        self.batch_size = batch_size
        self.params = calculate_dataloader_params(batch_size)

    def setup(self, stage=None):
        self.train_dataset = self.dataset(
            image_paths=self.train_data["paths"],
            labels=self.train_data["labels"],
            **self.dataset_args
        )
        self.val_dataset = ForestDataset(
            image_paths=self.val_data["paths"],
            labels=self.val_data["labels"]
        )
        self.test_dataset = ForestDataset(
            image_paths=self.test_data["paths"],
            labels=self.test_data["labels"]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **self.params)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, **self.params)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.params)


if __name__ == '__main__':
    params = calculate_dataloader_params(32)
    print(params)
