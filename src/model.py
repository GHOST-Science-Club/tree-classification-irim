import torch
from torch import nn
import pytorch_lightning as pl
from transforms import Transforms
from torchmetrics import Accuracy
import torchvision.models as models



class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3, transform=None):
        super(ResNetClassifier, self).__init__()
        self.save_hyperparameters()

        # Load data augmentation
        self.transform = transform

        # Load a pre-trained ResNet18 and modify the final layer
        self.model = models.resnet18(pretrained=True)

        # Freeze pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Define a loss function and metric
        self.criterion = nn.CrossEntropyLoss()
        if num_classes == 2:
            self.accuracy = Accuracy(task="binary")
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

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
        self.log('learning_rate', current_lr, prog_bar=True) # log learning rate for testing purposes
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

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.learning_rate)

        # Decay LR by a factor of 0.1 every 1 epoch
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1) # TODO: tune parameters

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }