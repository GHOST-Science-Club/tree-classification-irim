from pytorch_lightning.callbacks import Callback


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
        print(f"Epoch: {trainer.current_epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", end=' || ')

        self.val_metrics['loss'].append(val_loss)
        self.val_metrics['acc'].append(val_acc)
