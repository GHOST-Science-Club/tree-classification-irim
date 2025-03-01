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
    return ForestDataModule(
        sample_data, sample_data, sample_data, batch_size=2
        )


@pytest.mark.dataset
def test_dataset_length(dataset):
    error_msg = f"Dataset size is not as expected (size: {len(dataset)})"

    assert len(dataset) == 2, error_msg


@pytest.mark.dataset
def test_dataset_getitem(dataset):
    image, label = dataset[0]

    error_msg = {
        "not-tensor": f"Image is not a Tensor (type: {type(image)})",
        "not-rgb": f"Color channels are not RGB (channels: {image.shape[0]})",
        "label-str": f"Labels are of incorrect type (type: {type(label)})"
    }

    assert isinstance(image, torch.Tensor), error_msg["not-tensor"]
    # Ensure it has 3 color channels (RGB)
    assert image.shape[0] == 3, error_msg["not-rgb"]
    assert isinstance(label, int), error_msg["label-str"]


@pytest.mark.dataset
def test_transforms_applied(dataset):
    image, _ = dataset[0]

    error_msg = {
        "not-tensor": f"Image is not a Tensor (type: f{type(image)})",
        "invalid-min": f"Invalid min in norm (min: {torch.min(image)})",
        "invalid-max": f"Invalid max in norm (min: {torch.max(image)})"
    }

    assert isinstance(image, torch.Tensor), error_msg["not-tensor"]
    # Normalize check
    assert torch.min(image) >= -1.0, error_msg["invalid-min"]
    assert torch.max(image) <= 1.0, error_msg["invalid-max"]


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
def test_multiple_samples(dataset, sample_data):
    image1, label1 = dataset[0]
    image2, label2 = dataset[1]

    exp_label1 = sample_data['labels'][0]
    exp_label2 = sample_data['labels'][1]

    error_msg = {
        "invalid-size": f"Unexpected dataset size (size: {len(dataset)})",
        "not-tensor1": f"Image is not a Tensor (type: {type(image1)})",
        "not-tensor2": f"Image is not a Tensor (type: f{type(image2)})",
        "invalid-label1": f"Incorrect label. Expected label: {exp_label1}",
        "invalid-label2": f"Incorrect label. Expected label: {exp_label2}",
    }

    assert len(dataset) == 2, error_msg["invalid-size"]
    assert isinstance(image1, torch.Tensor), error_msg["not-tensor1"]
    assert isinstance(image2, torch.Tensor), error_msg["not-tensor2"]
    assert label1 == exp_label1, error_msg["invalid-label1"]
    assert label2 == exp_label2, error_msg["invalid-label2"]


@pytest.mark.dataset
def test_setup_creates_datasets(data_module):
    data_module.setup()

    train = data_module.train_dataset
    val = data_module.val_dataset
    test = data_module.test_dataset

    error_msg = {
        "train-invalid": "Train dataset in data module is invalid (None)",
        "val-invalid": "Validation dataset in data module is invalid (None)",
        "test-invalid": "Test dataset in data module is invalid (None)",
        "train-forest": "Train dataset is not ForestDataset",
        "val-forest": "Validation dataset is not ForestDataset",
        "test-forest": "Test dataset is not ForestDataset"
    }

    assert train is not None, error_msg["train-invalid"]
    assert val is not None, error_msg["val-invalid"]
    assert test is not None, error_msg["test-invalid"]
    assert isinstance(train, ForestDataset), error_msg["train-forest"]
    assert isinstance(val, ForestDataset), error_msg["val-forest"]
    assert isinstance(test, ForestDataset), error_msg["test-forest"]


@pytest.mark.dataset
def test_dataloader_returns_correct_batch_size(data_module):
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    images, labels = next(iter(train_loader))

    error_msg = {
        "train-loader": "Train data loader is not DataLoader",
        "val-loader": "Validation data loader is not DataLoader",
        "test-loader": "Test data loader is not DataLoader",
        "size-img": f"Incorrect image batch size (current: {images.shape[0]})",
        "size-lbl": f"Incorrect label batch size (current: {labels.shape[0]})"
    }

    assert isinstance(train_loader, DataLoader), error_msg["train-loader"]
    assert isinstance(val_loader, DataLoader), error_msg["val-loader"]
    assert isinstance(test_loader, DataLoader), error_msg["test-loader"]
    # Batch size
    assert images.shape[0] == 2, error_msg["size-img"]
    assert labels.shape[0] == 2, error_msg["size-lbl"]


@pytest.mark.dataset
def test_dataloader_shuffling(data_module):
    data_module.setup()
    train_loader = data_module.train_dataloader()

    error_msg = "Data in train data loader in not shuffled"

    assert train_loader.sampler is not None, error_msg
