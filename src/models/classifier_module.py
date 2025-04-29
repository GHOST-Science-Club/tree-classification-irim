import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from models.model_factory import create_model


class ClassifierModule(pl.LightningModule):
    def __init__(self, model_name, num_classes, step_size, gamma, learning_rate=1e-3, weight_decay=0, transform=None, freeze=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, num_classes, freeze)
        self.transform = transform
        self.learning_rate = learning_rate
        self.name = model_name
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        # Define a loss function and metric
        self.criterion = nn.CrossEntropyLoss()
        if num_classes == 2:
            self.accuracy = Accuracy(task="binary")
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Container for predictions
        self.predictions = None
        # Container for targets
        self.targets = None

    def forward(self, x):
        out = self.model(x)
        # If it's a tuple (Inception), return it directly
        if isinstance(out, tuple):
            return out
        # If it has logits, return logits
        if hasattr(out, "logits"):
            return out.logits
        return out

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.transform:
            x = self.transform(x)
        return x, y

    def step(self, batch, stage):
        images, labels = batch
        labels = labels.long()

        if stage == "train" and self.name.startswith("inception"):
            outputs, aux_outputs = self(images)
            loss1 = self.criterion(outputs, labels)
            loss2 = self.criterion(aux_outputs, labels)
            # inception specific loss
            loss = loss1 + 0.4 * loss2
        else:
            outputs = self(images)
            loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        if stage == "test":
            probs = torch.softmax(outputs, dim=1)
            self.predictions = torch.cat([self.predictions, probs], dim=0) if self.predictions is not None else probs
            self.targets = torch.cat([self.targets, labels], dim=0) if self.targets is not None else labels

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        if self.name.startswith("efficientnet"):
            optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }
