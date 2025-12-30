import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from models.model_factory import create_model
from models.diversified_model import GradientBoostingLoss


class ClassifierModule(pl.LightningModule):
    def __init__(self, model_name, num_classes, step_size, gamma, learning_rate=1e-3, weight_decay=0, freeze=False, weight=None,
                 optimizer="adam", momentum=0.9, scheduler="step", warmup_epochs=0, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, num_classes, freeze)
        self.learning_rate = learning_rate
        self.name = model_name
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.scheduler_name = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Define a loss function and metric

        if model_name == "fine_grained":
            self.criterion = GradientBoostingLoss()
        else:
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
        if self.name == "fine_grained":
            out = self.model(x, is_train=True)
        else:
            out = self.model(x)

        # If it's a tuple (Inception), return it directly
        if isinstance(out, tuple):
            return out[0]
        # If it has logits, return logits
        if hasattr(out, "logits"):
            return out.logits
        return out

    def step(self, batch, stage):
        images, labels = batch
        labels = labels.long()
        is_training = stage == "train"

        if is_training and self.name.startswith("inception"):
            outputs, aux_outputs = self.model(images)
            loss1 = self.criterion(outputs, labels)
            loss2 = self.criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        elif self.name == "fine_grained":
            outputs = self.model(images, is_train=is_training)
            loss = self.criterion(outputs, labels)
        elif self.name == "sim_trans":
            outputs, mfb_features = self.model(images)
            loss = self.criterion(outputs, labels)
        else:
            outputs = self.model(images)
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
        # Select parameters to optimize
        if self.name.startswith("efficientnet"):
            params = self.model.classifier.parameters()
        elif self.name in ("fine_grained", "sim_trans"):
            params = self.model.parameters()
        else:
            params = self.model.fc.parameters()
        
        # Create optimizer based on config
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params, 
                lr=self.hparams.learning_rate, 
                momentum=self.momentum,
                weight_decay=self.hparams.weight_decay
            )
        else:  # adam
            optimizer = torch.optim.Adam(
                params, 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay
            )

        # Create scheduler based on config
        if self.scheduler_name == "cosine":
            # Cosine annealing with warmup (SIM-Trans paper approach)
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            if self.warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer, 
                    start_factor=0.01, 
                    end_factor=1.0, 
                    total_iters=self.warmup_epochs
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=self.max_epochs - self.warmup_epochs
                )
                scheduler = SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, cosine_scheduler], 
                    milestones=[self.warmup_epochs]
                )
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        else:  # step
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.step_size, 
                gamma=self.gamma
            )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
