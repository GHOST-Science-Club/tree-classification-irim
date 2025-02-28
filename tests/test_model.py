import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.model import ResNetClassifier
from src.dataset import ForestDataset
from src.transforms import Preprocess

@pytest.fixture
def model(monkeypatch):
    model = ResNetClassifier(num_classes=2, learning_rate=1e-3, freeze=True)
    
    model.trainer = Trainer() #  running training_step() from pl.LightningModule throws an error without a trainer
    for _ in range(10): # if no optimizer is present, it triggers an index error as model retrieves current_lr for logs
        model.trainer.optimizers.append(torch.optim.Adam(model.model.fc.parameters()))
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None) # log is not tested at it is done by pytorch-lightning
        
    return model


@pytest.fixture
def sample_batch():
    """Fixture to create a sample batch of images and labels."""
    images = torch.randn(4, 3, 224, 224)  # Simulating a batch of 4 RGB images of size 224x224
    labels = torch.randint(0, 2, (4,))  # Random binary labels
    return images, labels


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
def data_loader(sample_data):
    """Fixture to create a DataLoader for testing training and validation steps."""
    dataset = ForestDataset(sample_data["paths"], sample_data["labels"], transform=Preprocess())
    return DataLoader(dataset, batch_size=2)


@pytest.mark.model
def test_model_initialization(model):
    
    assert model.model is not None, "Model is not initialized"
    # May need to be changed in the future for other models
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), f"Loss function is not cross-entropy. Current loss function: {model.criterion}"
    assert model.accuracy is not None, "Accuracy is not initialized"


@pytest.mark.model
@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_model_different_class_sizes(num_classes):

    model = ResNetClassifier(num_classes=num_classes)
    assert model.model.fc.out_features == num_classes, "Model classes do not match expected number of classes"


@pytest.mark.model
def test_forward_pass(model, sample_batch):
    images, _ = sample_batch
    outputs = model(images)
    
    # Expecting 4 samples and 2 output classes
    assert outputs.shape == (4, 2), "Forward pass produces wrong number of samples and classes"


@pytest.mark.model
def test_training_step(model, sample_batch):
    batch = sample_batch
    
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
    
    assert all(p is False for p in frozen_params[:-2])


@pytest.mark.model
def test_train_dataloader(model, data_loader):
    batch = next(iter(data_loader))
    loss = model.training_step(batch, batch_idx=0)

    assert loss.item() > 0
