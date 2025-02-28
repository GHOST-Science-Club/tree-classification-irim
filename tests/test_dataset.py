import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from src.dataset import ForestDataset, ForestDataModule


@pytest.fixture
def sample_data(tmp_path):
    """Creates data instance with two sample images."""
    image_path1 = tmp_path / "sample_image1.jpg"
    image1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
    image1.save(image_path1)
    
    image_path2 = tmp_path / "sample_image2.jpg"
    image2 = Image.new("RGB", (224, 224), color=(255, 225, 0))
    image2.save(image_path2)

    return {
        "paths": [image_path1, image_path2],
        "labels": [0, 1]
    }
    

@pytest.fixture
def dataset(sample_data):
    """Creates a dataset instance with two sample images."""
    return ForestDataset(sample_data["paths"], sample_data["labels"])


@pytest.fixture
def data_module(sample_data):
    return ForestDataModule(sample_data, sample_data, sample_data, batch_size=2)


@pytest.mark.dataset
def test_dataset_length(dataset):
    assert len(dataset) == 2, f"Dataset size is not as expected (size: {len(dataset)})"


@pytest.mark.dataset
def test_dataset_getitem(dataset):
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor), f"Image is not a Tensor (type: f{type(image)})"
    # Ensure it has 3 color channels (RGB)
    assert image.shape[0] == 3, f"Color channels are not RGB (channels: {image.shape[0]})"
    assert isinstance(label, int), f"Labels are of incorrect type (type: {type(label)})"


@pytest.mark.dataset
def test_transforms_applied(dataset):    
    image, _ = dataset[0]
    
    assert isinstance(image, torch.Tensor), f"Image is not a Tensor (type: f{type(image)})"
    # Normalize check
    assert torch.min(image) >= -1.0 and torch.max(image) <= 1.0, f"Invalid normalization (range: {torch.min(image)}-{torch.max(image)})"


@pytest.mark.dataset
def test_missing_file_handling(tmp_path):
    missing_image_path = tmp_path / "missing.jpg"
    missing_data = {"paths": [missing_image_path], "labels": [1]}
    dataset = ForestDataset(*missing_data.values())
    
    data_module = ForestDataModule(missing_data, missing_data, missing_data)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
        
    with pytest.raises(FileNotFoundError):
        data_module.setup()
        train_loader = data_module.train_dataloader()
        _ = next(iter(train_loader)) 


@pytest.mark.dataset
def test_multiple_samples(dataset):
    assert len(dataset) == 2, f"Dataset size is not as expected (size: {len(dataset)})"
    image1, label1 = dataset[0]
    image2, label2 = dataset[1]

    assert isinstance(image1, torch.Tensor), f"Image is not a Tensor (type: f{type(image1)})"
    assert isinstance(image2, torch.Tensor), f"Image is not a Tensor (type: f{type(image2)})"
    assert label1 == 0, f"Incorrect label. Expected label: 0"
    assert label2 == 1, f"Incorrect label. Expected label: 1"


@pytest.mark.dataset
def test_setup_creates_datasets(data_module):
    data_module.setup()

    assert data_module.train_dataset is not None, "Train dataset in data module is invalid (None)"
    assert data_module.val_dataset is not None, "Validation dataset in data module is invalid (None)"
    assert data_module.test_dataset is not None, "Test dataset in data module is invalid (None)"
    assert isinstance(data_module.train_dataset, ForestDataset), "Train dataset is not ForestDataset"
    assert isinstance(data_module.val_dataset, ForestDataset), "Validation dataset is not ForestDataset"
    assert isinstance(data_module.test_dataset, ForestDataset), "Test dataset is not ForestDataset"


@pytest.mark.dataset
def test_dataloader_returns_correct_batch_size(data_module):
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    assert isinstance(train_loader, DataLoader), "Train data loader is not DataLoader"
    assert isinstance(val_loader, DataLoader), "Validation data loader is not DataLoader"
    assert isinstance(test_loader, DataLoader), "Test data loader is not DataLoader"
    
    images, labels = next(iter(train_loader))    
    
    # Batch size
    assert images.shape[0] == 2, f"Incorrect batch size for images (current: {images.shape[0]})"
    assert labels.shape[0] == 2, f"Incorrect batch size for labels (current: {labels.shape[0]})"


@pytest.mark.dataset
def test_dataloader_shuffling(data_module):
    data_module.setup()
    train_loader = data_module.train_dataloader()

    assert train_loader.sampler is not None, "Data in train data loader in not shuffled"

