from pytorch_lightning.callbacks import Callback
from dataset import CurriculumLearningDataset


class PrintMetricsCallback(Callback):
    def __init__(self):
        self.train_metrics = {"loss": [], "acc": []}
        self.val_metrics = {"loss": [], "acc": []}

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss'].item()
        train_acc = trainer.callback_metrics['train_acc'].item()
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        self.train_metrics['loss'].append(train_loss)
        self.train_metrics['acc'].append(train_acc)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics['val_loss'].item()
        val_acc = trainer.callback_metrics['val_acc'].item()
        print(f"Epoch: {trainer.current_epoch}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", end=' || ')

        self.val_metrics['loss'].append(val_loss)
        self.val_metrics['acc'].append(val_acc)


class CurriculumLearningCallback(Callback):
    def __init__(self, initial_ratio, step_size, class_order, labels):
        self.initial_ratio = initial_ratio
        self.step_size = step_size
        self.class_order = class_order

        self.class_indices = {}
        for idx, label in enumerate(labels):
            self.class_indices.setdefault(label, []).append(idx)

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        datamodule = trainer.datamodule

        current_step = int(current_epoch / self.step_size)

        indices = []
        labels = self.class_order[:(current_step + 1) * self.initial_ratio]
        for label in labels:
            indices.extend(self.class_indices[label])

        if datamodule.dataset != CurriculumLearningDataset:
            raise Exception(f"Curriculum learning callback is being used, but the dataset in the datamodule is of type: {type(datamodule.dataset)}")

        datamodule.dataset_args["indices"] = indices
        datamodule.setup()
        datamodule.reset_train_dataloader()
