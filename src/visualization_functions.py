from datetime import datetime
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_n_samples(dataset: dict, species_folders: dict, n_of_images: int = 5):
    titles = [f'Channel [{i}]' for i in range(4)]
    titles.extend(['Whole 4 channels', 'Channel [0,1,2]', 'Channel [1,2,3]', 'PIL conversion'])
    channels = [[0], [1], [2], [3], [0, 1, 2, 3], [0, 1, 2], [1, 2, 3], None]

    unique_classes = np.unique(dataset['train']['labels'])
    class_names = list(species_folders.keys())

    for label in unique_classes:

        already_displayed = []  # This object contains already picked samples' indices
        indices = np.where(dataset['train']['labels'] == label)[0]

        n_rows = n_of_images
        n_cols = 8
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 30))

        for i, ax_row in enumerate(axs):

            # To avoid displaying repetitive images
            unique = False
            while not unique:
                idx = choice(indices)  # picking random sample
                unique = True if idx not in already_displayed else False

            already_displayed.append(idx)

            # Reading samples
            img_path = dataset["train"]["paths"][idx]
            img = np.array(Image.open(img_path))
            label = class_names[dataset["train"]["labels"][idx]]

            fig.suptitle(f'Class name: {label}', fontsize=36)
            fig.tight_layout()

            for j, ax in enumerate(ax_row):
                ax.imshow(img[..., channels[j]] if channels[j] else Image.fromarray(img).convert('RGB'))
                if i == 0:
                    ax.set_title(titles[j], fontsize=20)
                ax.axis('off')

        plt.tight_layout()

        # Saving image
        path = Path.cwd() / "src" / "plots"
        path.mkdir(exist_ok=True)

        plt.savefig(path / f"{label}.png")
        plt.close()


def plot_metrics(train_metrics: dict, val_metrics: dict):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(train_metrics["loss"], label="Train accuracy")
    axs[0].plot(val_metrics["loss"], label="Val accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Losses")
    axs[0].legend()

    axs[1].plot(train_metrics["acc"], label="Train accuracy")
    axs[1].plot(val_metrics["acc"], label="Val accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Losses")
    axs[1].legend()

    plt.tight_layout()

    # Saving image with timestamp to avoid overwriting
    path = Path.cwd() / "src" / "plots"
    path.mkdir(exist_ok=True)
    plt.savefig(path / f"acc_loss_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
