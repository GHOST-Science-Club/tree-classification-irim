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

    class Trainer:
        def __init__(self):
            self.current_epoch = 1
            self.callback_metrics = {"train_loss": torch.tensor(0.2345), "train_acc": torch.tensor(0.9123), "val_loss": torch.tensor(0.3456), "val_acc": torch.tensor(0.8765)}

    trainer = Trainer()

    return trainer


@pytest.mark.callbacks
def test_on_train_epoch_end(callback, mock_trainer):
    callback.on_train_epoch_end(mock_trainer, None)
    train_loss = callback.train_metrics["loss"]
    train_acc = callback.train_metrics["acc"]

    error_msg = {
        "loss-numb": "Invalid loss metric number in training callback",
        "acc-numb": "Invalid accuracy metric number in training callback",
        "invalid-loss": "Invalid loss in training callback",
        "invalid-acc": "Invalid accuracy in training callback",
    }

    assert len(train_loss) == 1, error_msg["loss-numb"]
    assert len(train_acc) == 1, error_msg["acc-numb"]
    assert train_loss[0] == pytest.approx(0.2345), error_msg["invalid-loss"]
    assert train_acc[0] == pytest.approx(0.9123), error_msg["invalid-acc"]


@pytest.mark.callbacks
def test_on_validation_epoch_end(callback, mock_trainer):
    callback.on_validation_epoch_end(mock_trainer, None)
    val_loss = callback.val_metrics["loss"]
    val_acc = callback.val_metrics["acc"]

    error_msg = {
        "loss-numb": "Invalid loss metric number in validation callback",
        "acc-numb": "Invalid accuracy metric number in validation callback",
        "invalid-loss": "Invalid loss in validation callback",
        "invalid-acc": "Invalid accuracy in validation callback",
    }

    assert len(val_loss) == 1, error_msg["loss-numb"]
    assert len(val_acc) == 1, error_msg["acc-numb"]
    assert val_loss[0] == pytest.approx(0.3456), error_msg["invalid-loss"]
    assert val_acc[0] == pytest.approx(0.8765), error_msg["invalid-acc"]


@pytest.mark.callbacks
def test_callback_prints_correctly(callback, mock_trainer, capsys):
    callback.on_train_epoch_end(mock_trainer, None)
    callback.on_validation_epoch_end(mock_trainer, None)

    exp_out = {"train-out": "Train Loss: 0.2345, Train Acc: 0.9123", "val-out": "Epoch: 1, Val Loss: 0.3456, Val Acc: 0.8765"}

    error_msg = {"train-msg": "Incorrect training callback output", "val-msg": "Incorrect validation callback output"}

    captured = capsys.readouterr()
    assert exp_out["train-out"] in captured.out, error_msg["train-msg"]
    assert exp_out["val-out"] in captured.out, error_msg["val-msg"]
