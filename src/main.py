import os
from pathlib import Path

import kornia.augmentation as kaug
import torch
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from callbacks import PrintMetricsCallback
from dataset import ForestDataModule
from dataset_functions import download_data, load_dataset
from git_functions import get_git_branch, generate_short_hash
from model import ResNetClassifier
from visualization_functions import (show_n_samples, plot_metrics,
                                     get_confusion_matrix,
                                     get_precision_recall_curve,
                                     get_roc_auc_curve)


def main():
    # Load configuration file
    with open("src/config.yaml", "r") as c:
        config = yaml.safe_load(c)

    # Create a dedicated folder for the PureForest dataset to keep each tree species
    # organized, avoiding multiple directories in the main content folder.
    dataset_folder = Path.cwd() / config["dataset"]["folder"]
    dataset_folder.mkdir(exist_ok=True)

    species_folders = config["dataset"]["species_folders"]
    main_subfolders = config["dataset"]["main_subfolders"]

    # =========================== DATA LOADING AND PREPROCESSING ================================== #

    download_data(species_folders, main_subfolders, dataset_folder)
    dataset, label_map = load_dataset(dataset_folder, species_folders)
    show_n_samples(dataset, species_folders)

    # =========================== INITIALIZING DATA AND MODEL ================================== #
    batch_size = config["training"]["batch_size"]
    num_classes = len(label_map)
    learning_rate = config["training"]["learning_rate"]
    transforms = kaug.Resize(size=(224, 224))
    freeze = config["training"]["freeze"]

    datamodule = ForestDataModule(
        dataset['train'], dataset['val'], dataset['test'], batch_size=batch_size
    )

    print(datamodule)

    model = ResNetClassifier(
        num_classes=num_classes,
        learning_rate=learning_rate,
        transform=transforms,
        freeze=freeze
    )

    # ====================================== TRAINING ========================================== #
    max_epochs = config["training"]["max_epochs"]
    device = config["device"] if torch.cuda.is_available() else "cpu"
    callbacks = [PrintMetricsCallback()]

    branch_name = get_git_branch()
    short_hash = generate_short_hash()
    run_name = f'{branch_name}-{short_hash}'

    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project="ghost-irim", name=run_name)
    wandb_logger = WandbLogger(
        name=run_name,
        project='ghost-irim',
        log_model=True
    )

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule)

    # ====================================== TESTING ========================================== #
    trainer.test(model, datamodule=datamodule)

    # Callbacks' service
    for callback in callbacks:
        if isinstance(callback, PrintMetricsCallback):
            train_metrics = callback.train_metrics
            val_metrics = callback.val_metrics
            plot_metrics(train_metrics, val_metrics)
            wandb.log({'Accuracy and Loss Curves': wandb.Image('src/plots/acc_loss_curves.png')})

    # Logging plots
    preds = model.predictions
    targets = model.targets

    get_confusion_matrix(preds, targets, class_names=list(label_map.keys()))
    get_roc_auc_curve(preds, targets, class_names=list(label_map.keys()))
    get_precision_recall_curve(preds, targets, class_names=list(label_map.keys()))

    filenames = ['confusion_matrix.png', 'precision_recall_curve.png', 'roc_auc_curve.png']
    titles = ['Confusion Matrix', 'Precision-Recall Curve', 'ROC AUC Curve']
    for filename, title in zip(filenames, titles):
        wandb.log({title: wandb.Image(f'src/plots/{filename}')})

    wandb.finish()


if __name__ == "__main__":
    main()
