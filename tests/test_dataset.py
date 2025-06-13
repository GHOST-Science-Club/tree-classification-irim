from typing import ClassVar

import pytest
import torch
from PIL import Image

# from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader
from src.dataset import ForestDataset, OversampledDataset, UndersampledDataset, ForestDataModule


@pytest.fixture
def missing_data(tmp_path):
    """Creates invalid data instance with one sample image."""
    missing_image_path = tmp_path / "missing.jpg"
    return {"paths": [missing_image_path], "labels": [1]}


@pytest.fixture
def forest_dataset(sample_data):
    """Creates a dataset instance with two sample images."""
    return ForestDataset(sample_data["paths"], sample_data["labels"])


@pytest.fixture
def oversampled_dataset(sample_data):
    """Create an oversampled dataset instance with two sample images."""
    return OversampledDataset(sample_data["paths"], sample_data["labels"])


@pytest.fixture
def undersampled_dataset(sample_data):
    """Create an undersampled dataset instance with two sample images."""
    return UndersampledDataset(sample_data["paths"], sample_data["labels"])


@pytest.fixture
def data_module(sample_data):
    """Create a data module instance with two sample images."""
    return ForestDataModule(sample_data, sample_data, sample_data, ForestDataset, batch_size=2)


@pytest.mark.dataset
@pytest.mark.parametrize("dataset", [ForestDataset, OversampledDataset, UndersampledDataset])
class TestDatasetsDataModule:
    """
    Create the same test setup for multiple classes handling dataset
    """

    error_msg: ClassVar[dict[str, str]] = {
        "not-tensor": "Image is not a Tensor",
        "not-rgb": "Color channels are not RGB",
        "label-str": "Incorrect label type",
        "invalid-min": "Invalid min in norm",
        "invalid-max": "Invalid max in norm",
        "train-invalid": "Train dataset in data module is invalid (None)",
        "val-invalid": "Validation dataset in data module is invalid (None)",
        "test-invalid": "Test dataset in data module is invalid (None)",
        "train-dataset": "Train dataset is incorrect",
        "val-forest": "Validation dataset is not ForestDataset",
        "test-forest": "Test dataset is not ForestDataset",
        "train-loader": "Train data loader is not DataLoader",
        "val-loader": "Validation data loader is not DataLoader",
        "test-loader": "Test data loader is not DataLoader",
        "size-img": "Incorrect image batch size",
        "size-lbl": "Incorrect label batch size",
        "shuffled": "Data in train data loader in not shuffled",
    }

    @pytest.fixture(autouse=True)
    def _create_test_setup(self, dataset, sample_data, missing_data, request):
        # request.getfixturevalue(
        # dataset setup
        self.dataset = dataset(sample_data["paths"], sample_data["labels"])

        # missing files setup
        self.dataset_missing_files = dataset(*missing_data.values())
        self.data_module_missing_data = ForestDataModule(missing_data, missing_data, missing_data, dataset, batch_size=2)

        # data module setup
        self.data_module = ForestDataModule(sample_data, sample_data, sample_data, dataset, batch_size=2)

    def test_dataset_getitem(self):
        image, label = self.dataset[0]

        assert isinstance(image, torch.Tensor), self.error_msg["not-tensor"]
        # Ensure it has 3 color channels (RGB)
        assert image.shape[0] == 3, self.error_msg["not-rgb"]
        assert isinstance(label, int), self.error_msg["label-str"]

    def test_transforms_applied(self):
        image, _ = self.dataset[0]

        assert isinstance(image, torch.Tensor), self.error_msg["not-tensor"]
        # Normalize check
        assert torch.min(image) >= -1.0, self.error_msg["invalid-min"]
        assert torch.max(image) <= 1.0, self.error_msg["invalid-max"]

    def test_missing_file_handling(self):
        with pytest.raises(FileNotFoundError):
            _ = self.dataset_missing_files[0]

        self.data_module_missing_data.setup()
        train_loader = self.data_module_missing_data.train_dataloader()

        with pytest.raises(FileNotFoundError):
            _ = next(iter(train_loader))

    def test_setup_creates_datasets(self, dataset):
        self.data_module.setup()

        train = self.data_module.train_dataset
        val = self.data_module.val_dataset
        test = self.data_module.test_dataset

        assert train is not None, self.error_msg["train-invalid"]
        assert val is not None, self.error_msg["val-invalid"]
        assert test is not None, self.error_msg["test-invalid"]
        assert isinstance(train, dataset), self.error_msg["train-dataset"]
        assert isinstance(val, ForestDataset), self.error_msg["val-forest"]
        assert isinstance(test, ForestDataset), self.error_msg["test-forest"]

    def test_dataloader_returns_correct_batch_size(self):
        self.data_module.setup()

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()

        images, labels = next(iter(train_loader))

        assert isinstance(train_loader, DataLoader), self.error_msg["train-loader"]
        assert isinstance(val_loader, DataLoader), self.error_msg["val-loader"]
        assert isinstance(test_loader, DataLoader), self.error_msg["test-loader"]
        # Batch size
        assert images.shape[0] == 2, self.error_msg["size-img"]
        assert labels.shape[0] == 2, self.error_msg["size-lbl"]

    def test_dataloader_shuffling(self):
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()

        assert train_loader.sampler is not None, self.error_msg["shuffled"]


@pytest.mark.dataset
def test_forest_dataset_length(forest_dataset):
    error_msg = f"Unexpected dataset size (size: {len(forest_dataset)})"

    assert len(forest_dataset) == 10, error_msg


@pytest.mark.dataset
def test_multiple_samples(forest_dataset, sample_data):
    image1, label1 = forest_dataset[0]
    image2, label2 = forest_dataset[1]

    exp_label1 = sample_data["labels"][0]
    exp_label2 = sample_data["labels"][1]

    length = len(forest_dataset)

    error_msg = {
        "invalid-size": f"Unexpected dataset size (size: {length})",
        "not-tensor1": f"Image is not a Tensor (type: {type(image1)})",
        "not-tensor2": f"Image is not a Tensor (type: {type(image2)})",
        "invalid-label1": f"Incorrect label. Expected label: {exp_label1}",
        "invalid-label2": f"Incorrect label. Expected label: {exp_label2}",
    }

    assert length == 10, error_msg["invalid-size"]
    assert isinstance(image1, torch.Tensor), error_msg["not-tensor1"]
    assert isinstance(image2, torch.Tensor), error_msg["not-tensor2"]
    assert label1 == exp_label1, error_msg["invalid-label1"]
    assert label2 == exp_label2, error_msg["invalid-label2"]


def mock_transform(x):
    """Create simple tranformation for testing."""
    return x * 2


def mock_minority_transform(x):
    """
    Create simple minority transform that increments
    the input for test verification.
    """
    return x + 1


@pytest.mark.dataset
def test_no_oversampling_needed():
    image_paths = [f"img_{i}.jpg" for i in range(10)]
    labels = [0] * 5 + [1] * 5  # All classes have more samples than threshold

    dataset = OversampledDataset(image_paths, labels, transform=mock_transform, oversample_threshold=2)

    assert len(dataset) == 10, "Dataset size changed."


@pytest.mark.dataset
def test_all_classes_are_minority():
    image_paths = [f"img_{i}.jpg" for i in range(6)]
    # All classes have only 2 samples (below threshold)
    labels = [0, 1, 2, 0, 1, 2]

    dataset = OversampledDataset(image_paths, labels, transform=mock_transform, minority_transform=mock_minority_transform, oversample_threshold=3, oversample_factor=2)

    assert len(dataset) > 6, "Dataset size didn't change."


@pytest.mark.dataset
def test_no_minority_transform(tmp_path):
    """Ensure dataset functions correctly when `minority_transform` is None."""
    image_paths = []
    for i in range(7):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (224, 224), color=(255, 0, 0)).save(img_path)

        image_paths.append(img_path)

    print(image_paths)
    labels = [0, 1, 0, 1, 2, 2, 3]  # Class 3 is a minority

    dataset = OversampledDataset(image_paths, labels, oversample_threshold=3, oversample_factor=2)

    image, label = dataset[0]
    print("lol", image)
    assert isinstance(image, torch.Tensor), "Image is not a tensor"
    assert isinstance(label, int), "Invalid label type"


@pytest.mark.dataset
def test_single_class_dataset(tmp_path):
    image_paths = []
    for i in range(5):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (224, 224), color=(255, 0, 0)).save(img_path)

        image_paths.append(img_path)
    labels = [0] * 5  # Only one class

    dataset = OversampledDataset(image_paths, labels, transform=mock_transform, oversample_threshold=3, oversample_factor=2)

    assert len(dataset) == 5, "Dataset size didn't change"
    assert all(label == 0 for _, label in dataset), "Labels are different"


@pytest.mark.dataset
def test_exact_threshold_behavior():
    image_paths = [f"img_{i}.jpg" for i in range(6)]
    labels = [0, 0, 0, 1, 1, 1]  # Both classes exactly at threshold

    dataset = OversampledDataset(image_paths, labels, transform=mock_transform, minority_transform=mock_minority_transform, oversample_threshold=3, oversample_factor=2)

    assert len(dataset) == 12, "Dataset size changed"


# TODO: Write tests the undersample classs and for checking
# TODO: transformations in oversample dataset
# @pytest.mark.dataset
# def test_transform_is_applied(oversampled_dataset, sample_data):
#     """Ensure that the standard `transform` is applied to every image."""
#     oversampled_dataset.transform = mock_transform
#     image, _ = oversampled_dataset[0]

#     with Image.open(sample_data["paths"][0]) as img:
#         img = pil_to_tensor(img)
#         exp_image = img * 2

#     print("Lol1", image, "Lol2", exp_image)

#     assert torch.equal(image, exp_image),
# "Transform didn't modify every image."
