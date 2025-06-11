import math
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


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

    axs[0].plot(train_metrics["loss"], label="Train loss")
    axs[0].plot(val_metrics["loss"][:-1], label="Val loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Losses")
    axs[0].legend()

    axs[1].plot(train_metrics["acc"], label="Train accuracy")
    axs[1].plot(val_metrics["acc"][:-1], label="Val accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    plt.tight_layout()

    path = Path.cwd() / "src" / "plots"
    path.mkdir(exist_ok=True)
    plt.savefig(path / "acc_loss_curves.png")
    plt.close()


def get_confusion_matrix(model_output, targets, filepath=None, class_names=None,
                         title="Confusion Matrix", show=False):
    """
    Generates a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for the given model output and targets and saves it in filepath.

    Parameters
    ----------
    model_output : torch.Tensor
        The output predictions from the model, expected to be of shape (N, C) where N is the batch size and C is the number of classes.
        Each row contains probabilities for each class.
    targets : torch.Tensor
        The ground truth labels, expected to be of shape (N,).
    filepath : str, default=Path.cwd() / "src" / "plots"
        If not None, the plot will be saved to this filepath. By default, it will save the plot in src/plots folder.
    class_names : list of str, optional
        List of class names corresponding to the classes. If provided, the length should match the number of classes in model_output.
    title : str, optional, default="Confusion Matrix"
        Title for the confusion matrix plot.
    show : bool, optional, default=False
        If True, the plot will be displayed in a window.

    Returns
    --------
    None

    Raises
    ------
    AssertionError
        If the dimensions of model_output and targets do not match the expected shapes.
    AssertionError
        If class_names is provided and its length does not match the number of classes in model_output.

    Examples
    --------
    # >>> output = torch.rand((50, 10))  # Probabilities for 10 classes
    # >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    # >>> get_confusion_matrix(output, targets, "plot.png", class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], show=True)
    """

    if filepath is None:
        filepath = Path.cwd() / "src" / "plots"

    assert model_output.shape[0] == targets.shape[0], "model_output and targets must have the same number of examples."
    assert len(targets.shape) == 1, "targets must be a 1D tensor."
    assert len(model_output.shape) == 2, "model_output must be a 2D tensor with shape (N, C)."

    num_classes = model_output.shape[1]

    if class_names is not None:
        assert len(class_names) == num_classes, "Length of class_names must match the number of classes in model_output."

    # Move tensors to CPU
    targets = targets.cpu()
    model_output = model_output.cpu()

    # Get predicted classes (argmax over probabilities)
    predicted_classes = torch.argmax(model_output, dim=1)

    # Compute confusion matrix
    matrix = confusion_matrix(targets.numpy(), predicted_classes.numpy())

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    ax.set_ylabel('Actual', size=12, labelpad=10)
    ax.set_xlabel('Predicted', size=12, labelpad=10)

    if title is not None:
        ax.set_title(title, pad=30)
        ax.title.set_fontsize(16)

    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Make sure that long label names are also visible
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('left')
    else:
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(np.arange(num_classes))
        ax.set_yticklabels(np.arange(num_classes))

    ax.xaxis.set_ticks_position('top')

    fig.tight_layout()

    if filepath is not None:
        filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath / "confusion_matrix.png")

    if show:
        plt.show()
    else:
        plt.close()


def get_roc_auc_curve(model_output, targets, filepath=None, class_of_interest=None,
                      class_names=None, title="ROC Curves", show=False):
    """
    Generates [ROC AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
    curves for a multi-class classification task and saves them to filepath.

    It uses the [OvR](https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a)
    (One versus Rest) approach - it compares each class against all the others at the same time.
    For every class it treats the model's outputs as outputs to a binary classification task. One class is considered as our "positive" class
    while all other classes are considered as one "negative" class.

    Parameters
    ----------
    model_output : torch.Tensor
         The output probabilities from the model, expected to be of shape (N, C) where N is the batch size and C is the number of classes.
    targets : torch.Tensor
        The ground truth labels, expected to be of shape (N,).
    filepath : str, default=Path.cwd() / "src" / "plots"
        If not None, the plot will be saved to this filepath. By default, it will save the plot in src/plots folder.
    class_of_interest : str, optional
        If class_of_interest is not None then only one plot gets generated for the given class of interest. Otherwise, it plots ROC AUC curves
        for every class in one figure. If provided, the class_names parameter must not be None.
    class_names : list of str, optional
        List of class names corresponding to the classes. If provided, the length should match the number of classes in model_output.
    title : str, optional, default="ROC Curves"
        Title for the ROC AUC plot.
    show : bool, optional, default=False
        If True, the plot will be displayed in a window.

    Returns
    --------
    None

    Raises
    ------
    AssertionError
        If the shapes of model_output and targets do not match the expected dimensions.
    AssertionError
        If class_names is provided and its length does not match the number of classes in model_output.
    AssertionError
        If class_of_interest is provided and class_names is None.

    Examples
    --------
    Without providing the class_of_interest:
    # >>> output = torch.rand((50, 10))
    # >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    # >>> get_roc_auc_curve(output, targets, "plot.png", show=True, class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

    With class_of_interest:
    # >>> output = torch.rand((50, 10))
    # >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    # >>> get_roc_auc_curve(output, targets, "plot.png", show=True, class_of_interest="b", class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    """

    if filepath is None:
        filepath = Path.cwd() / "src" / "plots"

    assert len(model_output.shape) == 2, "model_output must be a 2D tensor with shape (N, C)."
    assert len(targets.shape) == 1, "targets must be a 1D tensor."
    assert model_output.shape[0] == targets.shape[0], "model_output and targets must have the same number of samples."

    num_classes = model_output.shape[1]
    if class_names is not None:
        assert len(class_names) == num_classes, "class_names must have the same length as the number of classes in model_output."

    # Move the tensors to CPU
    targets = targets.cpu()
    model_output = model_output.cpu()

    if class_of_interest is not None:
        assert class_names is not None, "If class_of_interest is specified, class_names must not be None."
        assert class_of_interest in class_names, "class_of_interest must be in class_names."

    # One-vs-Rest approach
    if class_of_interest is None:
        n_of_rows = math.ceil(num_classes ** 0.5)
        n_of_cols = math.ceil(num_classes / n_of_rows)

        fig, ax = plt.subplots(n_of_rows, n_of_cols, figsize=(3 * n_of_cols, 3 * n_of_rows))
        ax = ax.flatten()

        for idx in range(num_classes):
            fpr, tpr, _ = roc_curve((targets == idx).numpy(), model_output[:, idx].numpy())
            roc_auc = auc(fpr, tpr)

            ax[idx].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

            if class_names is not None:
                subplot_title = class_names[idx]
            else:
                subplot_title = f"Class {idx}"

            ax[idx].set_title(subplot_title)
            ax[idx].set_ylabel('True Positive Rate')
            ax[idx].set_xlabel('False Positive Rate')
            ax[idx].legend(loc="lower right")

        # Hide unused subplots
        for i in range(num_classes, len(ax)):
            fig.delaxes(ax[i])

    else:
        class_num = class_names.index(class_of_interest)

        fig, ax = plt.subplots(figsize=(5, 5))

        fpr, tpr, _ = roc_curve((targets == class_num).numpy(), model_output[:, class_num].numpy())
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

        ax.set_title(class_of_interest)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.legend(loc="lower right")

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath / "roc_auc_curve.png")

    if show:
        plt.show()
    else:
        plt.close()


def get_precision_recall_curve(model_output, targets, filepath=None, class_of_interest=None,
                               class_names=None, title="Precision Recall Curves", show=False):
    """
    Generates precision-recall curves for a multi-class classification task and saves them to filepath.

    It uses the One-vs-Rest (OvR) approach, comparing each class against all the others at the same time.
    For every class, it treats the model's outputs as outputs to a binary classification task. One class is considered as our "positive" class
    while all other classes are considered as one "negative" class.

    Parameters
    ----------
    model_output : torch.Tensor
        The output predictions from the model, expected to be of shape (N, C) where N is the batch size and C is the number of classes.
        Each row corresponds to a vector of probabilities for each class.
    targets : torch.Tensor
        The ground truth labels, expected to be of shape (N,).
    filepath : str, default=Path.cwd() / "src" / "plots"
        If not None, the plot will be saved to this filepath. By default, it will save the plot in src/plots folder.
    class_of_interest : str, optional
        If class_of_interest is not None then only one plot gets generated for the given class of interest. Otherwise, it plots precision
        recall curves for every class in one figure. If provided, the class_names parameter must not be None.
    class_names : list of str, optional
        List of class names corresponding to the classes. If provided, the length should match the number of classes in model_output.
    title : str, optional, default="Precision Recall Curves"
        Title for the precision-recall plot.
    show : bool, optional, default=False
        If True, the plot will be displayed in a window.

    Returns
    --------
    None

    Raises
    ------
    AssertionError
        If the shapes of model_output and targets do not match the expected dimensions.
    AssertionError
        If class_names is provided and its length does not match the number of classes in model_output.
    AssertionError
        If class_of_interest is provided and class_names is None.
    """

    if filepath is None:
        filepath = Path.cwd() / "src" / "plots"

    assert model_output.ndim == 2, "model_output must be a 2D tensor with shape (N, C)."
    assert targets.ndim == 1, "targets must be a 1D tensor with shape (N,)."
    assert model_output.size(0) == targets.size(0), "model_output and targets must have the same number of samples."

    # Move tensors to CPU
    targets = targets.cpu()
    model_output = model_output.cpu()

    num_classes = model_output.size(1)

    if class_names is not None:
        assert len(class_names) == num_classes, "Length of class_names must match the number of classes in model_output."

    if class_of_interest is not None:
        assert class_names is not None, "If class_of_interest is specified, class_names must not be None."
        assert class_of_interest in class_names, "class_of_interest must be in class_names."

    # One-vs-Rest approach
    if class_of_interest is None:
        n_of_rows = math.ceil(num_classes ** 0.5)
        n_of_cols = math.ceil(num_classes / n_of_rows)

        fig, ax = plt.subplots(n_of_rows, n_of_cols, figsize=(3 * n_of_cols, 3 * n_of_rows))
        ax = ax.flatten()

        for idx in range(num_classes):
            precision, recall, _ = precision_recall_curve((targets == idx).numpy(), model_output[:, idx].numpy())

            ax[idx].plot(recall, precision)

            if class_names is not None:
                subplot_title = class_names[idx]
            else:
                subplot_title = f"Class {idx}"

            ax[idx].set_title(subplot_title)
            ax[idx].set_ylabel('Precision')
            ax[idx].set_xlabel('Recall')

        # Hide unused subplots
        for i in range(num_classes, len(ax)):
            fig.delaxes(ax[i])

    else:
        class_idx = class_names.index(class_of_interest)

        fig, ax = plt.subplots(figsize=(5, 5))

        precision, recall, _ = precision_recall_curve((targets == class_idx).numpy(), model_output[:, class_idx].numpy())

        ax.plot(recall, precision)

        ax.set_title(class_of_interest)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()

    if filepath is not None:
        filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath / "precision_recall_curve.png")

    if show:
        plt.show()
    else:
        plt.close()
