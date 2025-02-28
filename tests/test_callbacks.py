import pytest
import torch
from src.callbacks import PrintMetricsCallback

@pytest.fixture
def callback():
    """Create a fresh instance of PrintMetricsCallback for each test."""
    return PrintMetricsCallback()

@pytest.fixture
def mock_trainer():
    """Create a mock trainer with predefined callback metrics."""
    class Trainer():
        def __init__(self):
            self.current_epoch = 1
            self.callback_metrics = {
                'train_loss': torch.tensor(0.2345),
                'train_acc': torch.tensor(0.9123),
                'val_loss': torch.tensor(0.3456),
                'val_acc': torch.tensor(0.8765)
            }
    
    trainer = Trainer()
    
    return trainer


@pytest.mark.callbacks
def test_on_train_epoch_end(callback, mock_trainer):
    callback.on_train_epoch_end(mock_trainer, None)

    assert len(callback.train_metrics["loss"]) == 1, "Invalid loss metric number in training callback"
    assert len(callback.train_metrics["acc"]) == 1, "Invalid accuracy metric number in training callback"
    assert callback.train_metrics["loss"][0] == pytest.approx(0.2345), "Invalid loss in training callback"
    assert callback.train_metrics["acc"][0] == pytest.approx(0.9123), "Invalid accuracy in training callback"


@pytest.mark.callbacks
def test_on_validation_epoch_end(callback, mock_trainer):
    callback.on_validation_epoch_end(mock_trainer, None)

    assert len(callback.val_metrics["loss"]) == 1, "Invalid loss metric number in validation callback"
    assert len(callback.val_metrics["acc"]) == 1, "Invalid accuracy metric number in validation callback"
    assert callback.val_metrics["loss"][0] == pytest.approx(0.3456), "Invalid loss in validation callback"
    assert callback.val_metrics["acc"][0] == pytest.approx(0.8765), "Invalid accuracy in validation callback"


@pytest.mark.callbacks
def test_callback_prints_correctly(callback, mock_trainer, capsys):
    callback.on_train_epoch_end(mock_trainer, None)
    callback.on_validation_epoch_end(mock_trainer, None)

    captured = capsys.readouterr()
    assert "Train Loss: 0.2345, Train Acc: 0.9123" in captured.out, "Incorrect training callback output"
    assert "Epoch: 1, Val Loss: 0.3456, Val Acc: 0.8765" in captured.out, "Incorrect validation callback output"
