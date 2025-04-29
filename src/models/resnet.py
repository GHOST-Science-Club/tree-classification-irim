import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.models as models


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3, weight_decay=0, transform=None, freeze=False, weight=None):
        super(ResNetClassifier, self).__init__()
        self.save_hyperparameters()

        self.transform = transform
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = models.resnet18(weights='DEFAULT')

        # Freeze pre-trained layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Copy the red channel to infrared channel
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        with torch.no_grad():
            self.model.conv1.weight[:, 0] = old_conv.weight[:, 0]
            self.model.conv1.weight[:, 1:] = old_conv.weight

        # Define a loss function and metric
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        if num_classes == 2:
            self.accuracy = Accuracy(task="binary")
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Container for predictions
        self.predictions = None
        # Container for targets
        self.targets = None

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.transform:
            x = self.transform(x)
        return x, y

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate and log accuracy
        predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)  # log learning rate for testing purposes
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        predicted_classes = torch.argmax(outputs, dim=1)
        acc = self.accuracy(predicted_classes, labels)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        predicted_probs = torch.softmax(outputs, dim=1)

        # Save predictions and targets for later use
        if self.predictions is None:
            self.predictions = predicted_probs
            self.targets = labels
        else:
            self.predictions = torch.cat((self.predictions, predicted_probs), dim=0)
            self.targets = torch.cat((self.targets, labels), dim=0)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=0.001, weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.hparams.learning_rate,
            max_lr=0.001,
            step_size_up=13630,
            mode="triangular2",
            cycle_momentum=False
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }
