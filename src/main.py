import torch
from pathlib import Path
import kornia.augmentation as kaug
from pytorch_lightning import Trainer


from model import ResNetClassifier
from dataset import ForestDataModule
from callbacks import PrintMetricsCallback
from visualization_functions import show_n_samples, plot_metrics
from dataset_functions import download_data, load_dataset, clip_balanced_dataset


def main():

    # Create a dedicated folder for the PureForest dataset to keep each tree species
    # organized, avoiding multiple directories in the main content folder.
    dataset_folder = Path.cwd() / "src" / "data"
    dataset_folder.mkdir(exist_ok=True)

    species_folders = {
        "Castanea_sativa": "data/imagery-Castanea_sativa.zip",
        "Pinus_nigra": "data/imagery-Pinus_nigra.zip"
    }

    main_subfolders = {
        "aerial_imagery": "imagery/",
        "lidar": "lidar/"
    }


    # =========================== DATA LOADING AND PREPROCESSING ================================== #

    download_data(species_folders, main_subfolders, dataset_folder)
    dataset, label_map = load_dataset(dataset_folder)
    show_n_samples(dataset, species_folders)
    #clipped_dataset = clip_balanced_dataset(dataset)


    # =========================== INITIALIZING DATA AND MODEL ================================== #
    BATCH_SIZE = 32
    NUM_CLASSES = 2
    LEARNING_RATE = 0.001
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
    MAX_EPOCHS = 10
    DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
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

    train_metrics = CALLBACKS[0].train_metrics
    val_metrics = CALLBACKS[0].val_metrics

    plot_metrics(train_metrics, val_metrics)


if __name__ == "__main__":
    main()