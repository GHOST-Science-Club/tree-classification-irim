import pytest
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.models.resnet import ResNetClassifier
from src.dataset import ForestDataset
from src.transforms import Preprocess


@pytest.fixture
def model(monkeypatch):
    model = ResNetClassifier(num_classes=2, learning_rate=1e-3, freeze=True)

    # running training_step() from pl.LightningModule throws
    # an error without a trainer
    model.trainer = Trainer()

    # if no optimizer is present, it triggers an index error
    # as model retrieves current_lr for logs
    for _ in range(10):
        model.trainer.optimizers.append(
            torch.optim.Adam(model.model.fc.parameters())
        )

    # log is not tested at it is done by pytorch-lightning
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)

    return model


@pytest.fixture
def sample_batch():
    """Fixture to create a sample batch of images and labels."""

    # Simulating a batch of 4 RGB images of size 224x224
    images = torch.randn(4, 3, 224, 224)

    # Random binary labels
    labels = torch.randint(0, 2, (4,))

    return images, labels


@pytest.fixture
def data_loader(sample_data):
    """Fixture to create a DataLoader for
    testing training and validation steps."""
    dataset = ForestDataset(
        sample_data["paths"], sample_data["labels"]
    )
    return DataLoader(dataset, batch_size=2)


@pytest.mark.model
def test_model_initialization(model):

    error_msg = {
        "not-init-m": "Model is not initialized",
        "bad-loss": f"Unexpected loss function ({model.criterion})",
        "not-init-acc": "Accuracy is not initialized"
    }

    assert model.model is not None, error_msg["not-init-m"]
    # May need to be changed in the future for other models
    assert isinstance(
        model.criterion, torch.nn.CrossEntropyLoss
    ), error_msg["bad-loss"]
    assert model.accuracy is not None, error_msg["not-init-m"]


@pytest.mark.model
@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_model_different_class_sizes(num_classes):
    model = ResNetClassifier(num_classes=num_classes)

    error_msg = f"""
    Model classes ({model.model.fc.out_features}) do not match
    expected number of classes: {num_classes}.
    """

    assert model.model.fc.out_features == num_classes, error_msg


@pytest.mark.model
def test_forward_pass(model, sample_batch):
    images, _ = sample_batch
    outputs = model(images)

    error_msg = f"""
    Forward pass produces wrong number of
    samples and classes. Current: ({outputs.shape})
    """

    # Expecting 4 samples and 2 output classes
    assert outputs.shape == (4, 2), error_msg


@pytest.mark.model
def test_training_step(model, sample_batch):
    batch = sample_batch

    loss = model.training_step(batch, batch_idx=0)

    assert loss is not None, "Loss is invalid"
    assert loss.item() > 0, "Loss is less than 0"


@pytest.mark.model
def test_validation_step(model, sample_batch):
    batch = sample_batch

    loss = model.validation_step(batch, batch_idx=0)

    assert loss is not None, "Loss is invalid"
    assert loss.item() > 0, "Loss is less than 0"


@pytest.mark.model
def test_test_step(model, sample_batch):
    batch = sample_batch

    loss = model.test_step(batch, batch_idx=0)

    assert loss is not None, "Loss is invalid"
    assert loss.item() > 0, "Loss is less than 0"


@pytest.mark.model
def test_optimizer_configuration(model):
    optim_config = model.configure_optimizers()

    scheduler = optim_config["lr_scheduler"]["scheduler"]

    error_msg = {
        "no-opt": "configure_optimizer does not output optimizer",
        "invalid-opt": "Optimizer is invalid (None)",
        "no-lr": "configure_optimizer does not output learning scheduler",
        "invalid_lr": "Scheduler is invalid (None)"
    }

    assert "optimizer" in optim_config, error_msg["no-opt"]
    assert optim_config["optimizer"] is not None, error_msg["invalid-opt"]
    assert "lr_scheduler" in optim_config, error_msg["no-lr"]
    assert scheduler is not None, error_msg["invalid_lr"]


@pytest.mark.model
def test_model_freezing(model):
    frozen_params = [p.requires_grad for p in model.model.parameters()]

    error_msg = "Pretrained weights are not frozen"

    assert all(p is False for p in frozen_params[:-2]), error_msg


@pytest.mark.model
def test_train_dataloader(model, data_loader):
    batch = next(iter(data_loader))
    loss = model.training_step(batch, batch_idx=0)

    assert loss.item() > 0, "Dataloader produces loss less than 0"


@pytest.mark.model
def test_on_after_batch_transfer_without_transform(model, sample_batch):
    result = model.on_after_batch_transfer(sample_batch, dataloader_idx=0)

    error_msg = {
        "img": "Image has been altered by after batch transfer",
        "lbl": "Label has been altered by after batch transfer"
    }

    assert torch.equal(result[0], sample_batch[0]), error_msg["img"]
    assert torch.equal(result[1], sample_batch[1]), error_msg["lbl"]


@pytest.mark.model
def test_on_after_batch_transfer_with_transform(model, sample_batch):
    def mock_transform(x):
        return x * 2

    model.transform = mock_transform
    result = model.on_after_batch_transfer(sample_batch, dataloader_idx=0)

    expected_x = sample_batch[0] * 2

    error_msg = {
        "img": "Image remained unchanged after batch transfer",
        "lbl": "Label has been altered by after batch transfer"
    }

    assert torch.equal(result[0], expected_x), error_msg["img"]
    assert torch.equal(result[1], sample_batch[1]), error_msg["lbl"]
