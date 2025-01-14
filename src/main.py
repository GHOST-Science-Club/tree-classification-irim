import yaml
import torch
from pathlib import Path
import kornia.augmentation as kaug
from pytorch_lightning import Trainer


from model import ResNetClassifier
from dataset import ForestDataModule
from callbacks import PrintMetricsCallback
from dataset_functions import download_data, load_dataset
from visualization_functions import show_n_samples, plot_metrics


def main():

    # Load configuration file
    with open("src/config.yaml", "r") as c:
        CONFIG = yaml.safe_load(c)

    # Create a dedicated folder for the PureForest dataset to keep each tree species
    # organized, avoiding multiple directories in the main content folder.
    dataset_folder = Path.cwd() / CONFIG["dataset"]["folder"]
    dataset_folder.mkdir(exist_ok=True)

    species_folders = CONFIG["dataset"]["species_folders"]
    main_subfolders = CONFIG["dataset"]["main_subfolders"]


    # =========================== DATA LOADING AND PREPROCESSING ================================== #

    download_data(species_folders, main_subfolders, dataset_folder)
    dataset, label_map = load_dataset(dataset_folder, species_folders)
    show_n_samples(dataset, species_folders)

    # =========================== INITIALIZING DATA AND MODEL ================================== #
    BATCH_SIZE = CONFIG["training"]["batch_size"]
    NUM_CLASSES = CONFIG["training"]["num_classes"]
    LEARNING_RATE = CONFIG["training"]["learning_rate"]
    TRANSFORMS = kaug.Resize(size=(224, 224))

    datamodule = ForestDataModule(
        dataset['train'], dataset['val'], dataset['test'], batch_size=BATCH_SIZE
    )

    print(datamodule)

    model = ResNetClassifier(
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        transform=TRANSFORMS
    )

    # ====================================== TRAINING ========================================== #
    MAX_EPOCHS = CONFIG["training"]["max_epochs"]
    DEVICE = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    CALLBACKS = [PrintMetricsCallback()]

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        devices=1,
        callbacks=CALLBACKS
    )

    trainer.fit(model, datamodule)

    # ====================================== TESTING ========================================== #
    trainer.test(model, datamodule=datamodule)

    # Callbacks' service
    for callback in CALLBACKS:
        if isinstance(callback, PrintMetricsCallback):
            train_metrics = callback.train_metrics
            val_metrics = callback.val_metrics
            plot_metrics(train_metrics, val_metrics)


if __name__ == "__main__":
    main()