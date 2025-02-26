import pytest
import torch
from PIL import Image
from src.dataset import ForestDataset, ForestDataModule


@pytest.fixture
def sample_image(tmp_path):
    """Creates a temporary sample image for testing."""
    image_path = tmp_path / "sample_image.jpg"
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    image.save(image_path)
    return image_path


@pytest.fixture
def dataset(sample_image):
    """Creates a dataset instance with a sample image."""
    image_paths = [sample_image]
    labels = [0]  # Example label
    return ForestDataset(image_paths, labels)


@pytest.mark.dataset
def test_dataset_length(dataset):
    assert len(dataset) == 1, f"Dataset size is not as expected (size: {len(dataset)})"


@pytest.mark.dataset
def test_dataset_getitem(dataset):
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor), f"Image is not a Tensor (type: f{type(image)})"
    # Ensure it has 3 color channels (RGB)
    assert image.shape[0] == 3, f"Color channels are not RGB (channels: {image.shape[0]})"
    assert isinstance(label, int), f"Labels are of incorrect type (type: {type(label)})"


@pytest.mark.dataset
def test_transforms_applied(sample_image):
    dataset = ForestDataset([sample_image], [1])
    
    image, _ = dataset[0]
    assert isinstance(image, torch.Tensor), f"Image is not a Tensor (type: f{type(image)})"
    # Normalize check
    assert torch.min(image) >= -1.0 and torch.max(image) <= 1.0, f"Invalid normalization (range: {torch.min(image)}-{torch.max(image)})"


@pytest.mark.dataset
def test_missing_file_handling(tmp_path):
    missing_image_path = tmp_path / "missing.jpg"
    dataset = ForestDataset([missing_image_path], [1])

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


@pytest.mark.dataset
def test_multiple_samples(tmp_path):
    img1 = Image.new("RGB", (64, 64), color=(255, 255, 255))
    img2 = Image.new("RGB", (64, 64), color=(0, 0, 0))

    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    img1.save(img1_path)
    img2.save(img2_path)

    dataset = ForestDataset([img1_path, img2_path], [0, 1])
    
    assert len(dataset) == 2, f"Dataset size is not as expected (size: {len(dataset)})"
    image1, label1 = dataset[0]
    image2, label2 = dataset[1]

    assert isinstance(image1, torch.Tensor), f"Image is not a Tensor (type: f{type(image1)})"
    assert isinstance(image2, torch.Tensor), f"Image is not a Tensor (type: f{type(image2)})"
    assert label1 == 0, f"Incorrect label. Expected label: 0"
    assert label2 == 1, f"Incorrect label. Expected label: 1"
