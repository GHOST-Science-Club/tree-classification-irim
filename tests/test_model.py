import pytest
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.model import ResNetClassifier
from src.dataset import ForestDataset
from src.transforms import Preprocess

@pytest.fixture
def model():
    return ResNetClassifier(num_classes=2, learning_rate=1e-3, freeze=True)


@pytest.fixture
def sample_batch():
    """Fixture to create a sample batch of images and labels."""
    images = torch.randn(4, 3, 224, 224)  # Simulating a batch of 4 RGB images of size 224x224
    labels = torch.randint(0, 2, (4,))  # Random binary labels
    return images, labels


@pytest.fixture
def data_loader(sample_batch):
    """Fixture to create a DataLoader for testing training and validation steps."""
    dataset = ForestDataset(sample_batch[0], sample_batch[1], transform=Preprocess())
    return DataLoader(dataset, batch_size=2)


@pytest.mark.model
def test_model_initialization(model):
    
    assert model.model is not None
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.accuracy is not None


@pytest.mark.model
@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_model_different_class_sizes(num_classes):

    model = ResNetClassifier(num_classes=num_classes)
    assert model.model.fc.out_features == num_classes


@pytest.mark.model
def test_forward_pass(model, sample_batch):
    images, _ = sample_batch
    outputs = model(images)
    
    assert outputs.shape == (4, 2)  # Expecting 4 samples and 2 output classes


@pytest.mark.model
def test_training_step(monkeypatch, model, sample_batch):
    batch = sample_batch
    
    model.trainer = Trainer()
    
    for _ in range(10):
        model.trainer.optimizers.append(torch.optim.Adam(model.model.fc.parameters()))
    
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    # monkeypatch.setattr(model, "trainer.optimizers[0].param_groups[0]['lr']", lambda *args, **kwargs: "mock_trainer")
    # with mock.patch.object(model, "trainer.optimizers[0].param_groups[0]['lr']", lambda *args, **kwargs: "mock_trainer"):
    loss = model.training_step(batch, batch_idx=0)
    
    assert loss is not None
    assert loss.item() > 0


@pytest.mark.model
def test_validation_step(model, sample_batch):
    batch = sample_batch
    loss = model.validation_step(batch, batch_idx=0)
    
    assert loss is not None
    assert loss.item() > 0


@pytest.mark.model
def test_test_step(model, sample_batch):
    batch = sample_batch
    loss = model.test_step(batch, batch_idx=0)
    
    assert loss is not None
    assert loss.item() > 0


@pytest.mark.model
def test_optimizer_configuration(model):
    optim_config = model.configure_optimizers()
    
    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config
    assert optim_config["lr_scheduler"]["scheduler"] is not None


@pytest.mark.model
def test_model_freezing(model):
    frozen_params = [p.requires_grad for p in model.model.parameters()]
    
    assert all(p is False for p in frozen_params)


@pytest.mark.model
def test_train_dataloader(model, data_loader):
    for batch in data_loader:
        loss = model.training_step(batch, batch_idx=0)
        assert loss.item() > 0
        break
