import math
from datetime import datetime
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def get_confusion_matrix(model_output, targets, filepath, class_names=None, title="Confusion Matrix", show=False):
    """
    Generates a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for the given model output and targets and saves it in filepath.

    Parameters
    ----------
    model_output : torch.Tensor
        The output predictions from the model, expected to be of shape (N, C) where N is the batch size and C is the number of classes.
    targets : torch.Tensor
        The ground truth labels, expected to be of shape (N,).
    filepath : str
        If not None, the plot will be saved to this filepath.
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
        If the shapes of model_output and targets do not match the expected dimensions.
    AssertionError
        If class_names is provided and its length does not match the number of classes in model_output.

    Examples
    --------
    >>> output = torch.randn((50, 10), dtype=torch.float32)
    >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    >>> get_confusion_matrix(output, targets, "plot.png", class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], show=True)
    """

    assert model_output.shape[0] == targets.shape[0]
    assert len(targets.shape) == 1
    assert len(model_output.shape) == 2

    if class_names is not None:
        assert len(class_names) == model_output.shape[1]

    matrix = confusion_matrix(targets, torch.argmax(model_output, dim=1))

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
    else:
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(np.arange(10))
        ax.set_yticklabels(np.arange(10))

    ax.xaxis.set_ticks_position('top')

    if filepath is not None:
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()


def get_roc_auc_curve(model_output, targets, filepath, class_of_interest=None, class_names=None, title="ROC Curves",
                      show=False):
    """
    Generates [ROC AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
    curves for a multi-class classification task and saves them to filepath.

    It uses the [OvR](https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a)
    (One versus Rest) approach - it compares each class against all the others at the same time.
    For every class it treats the model's ouputs as outputs to a binary classification task. One class is considered as our "positive" class
    while all other classes are considered as one "negative" class.

    Parameters
    ----------
    model_output : torch.Tensor
        The output predictions from the model, expected to be of shape (N, C) where N is the batch size and C is the number of classes.
    targets : torch.Tensor
        The ground truth labels, expected to be of shape (N,).
    filepath : str
        If not None, the plot will be saved to this filepath.
    class_of_interest : str, optional
        If class_of_interest is not None then only one plot gets generated for the given class of interest. Otherwise it plots ROC AUC curves
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
        If class_of_interest is provided and but class_names is None.

    Examples
    --------
    Without providing the class_of_interest:
    >>> output = torch.randn((50, 10), dtype=torch.float32)
    >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    >>> get_roc_auc_curve(output, targets, "plot.png", show=True, class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

    With class_of_interest:
    >>> output = torch.randn((50, 10), dtype=torch.float32)
    >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    >>> get_roc_auc_curve(output, targets, "plot.png", show=True, class_of_interest="b", class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    """

    assert class_of_interest is None or class_names is not None, "If class_of_interest is provided class_names must not be None"
    assert model_output.shape[0] == targets.shape[0]
    assert len(targets.shape) == 1
    assert len(model_output.shape) == 2

    if class_names is not None:
        assert len(class_names) == model_output.shape[1]

    num_classes = model_output.shape[1]

    if class_of_interest is None:
        n_of_rows = math.ceil(num_classes ** 0.5)
        n_of_cols = math.ceil(num_classes / n_of_rows)

        fig, ax = plt.subplots(n_of_rows, n_of_cols, figsize=(3 * n_of_cols, 3 * n_of_rows))

        for class_num in range(num_classes):
            row_num = class_num // n_of_cols
            col_num = class_num % n_of_cols

            fpr, tpr, thresholds = roc_curve(F.one_hot(targets)[:, class_num], model_output[:, class_num])

            roc_auc = auc(fpr, tpr)

            ax[row_num, col_num].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

            ax[row_num, col_num].legend(loc="lower right")

            if class_names is not None:
                subplot_title = class_names[class_num]
            else:
                subplot_title = f"Class {class_num}"
            ax[row_num, col_num].set_title(subplot_title)

            ax[row_num, col_num].set_ylabel('True Positive Rate')
            ax[row_num, col_num].set_xlabel('False Positive Rate')

        # Hide unused subplots
        for i in range(num_classes, n_of_rows * n_of_cols):
            row_num = i // n_of_cols
            col_num = i % n_of_cols
            fig.delaxes(ax[row_num, col_num])
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

        class_num = class_names.index(class_of_interest)

        fpr, tpr, thresholds = roc_curve(F.one_hot(targets)[:, class_num], model_output[:, class_num])

        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

        ax.legend(loc="lower right")

        if class_names is not None:
            subplot_title = class_names[class_num]
        else:
            subplot_title = f"Class {class_num}"

        ax.set_title(subplot_title)

        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()


def get_precision_recall_curve(model_output, targets, filepath, class_of_interest=None, class_names=None,
                               title="Precision Recall Curves", show=False):
    """
    Generates [precision recall](https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248)
    curves for a multi-class classification task and saves them to filepath.

    It uses the [OvR](https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a)
    (One versus Rest) approach - it compares each class against all the others at the same time.
    For every class it treats the model's ouputs as outputs to a binary classification task. One class is considered as our "positive" class
    while all other classes are considered as one "negative" class.

    Parameters
    ----------
    model_output : torch.Tensor
        The output predictions from the model, expected to be of shape (N, C) where N is the batch size and C is the number of classes.
    targets : torch.Tensor
        The ground truth labels, expected to be of shape (N,).
    filepath : str
        If not None, the plot will be saved to this filepath.
    class_of_interest : str, optional
        If class_of_interest is not None then only one plot gets generated for the given class of interest. Otherwise it plots precision
        recall curves for every class in one figure. If provided, the class_names parameter must not be None.
    class_names : list of str, optional
        List of class names corresponding to the classes. If provided, the length should match the number of classes in model_output.
    title : str, optional, default="Precision Recall Curves"
        Title for the precision recall plot.
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
        If class_of_interest is provided and but class_names is None.

    Examples
    --------
    Without providing the class_of_interest:
    >>> output = torch.randn((50, 10), dtype=torch.float32)
    >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    >>> get_precision_recall_curve(output, targets, "plot.png", show=True, class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

    With class_of_interest:
    >>> output = torch.randn((50, 10), dtype=torch.float32)
    >>> targets = torch.randint(0, 10, (50,), dtype=torch.int64)
    >>> get_precision_recall_curve(output, targets, "plot.png", show=True, class_of_interest="b", class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    """

    assert class_of_interest is None or class_names is not None, "If class_of_interest is provided class_names must not be None"
    assert model_output.shape[0] == targets.shape[0]
    assert len(targets.shape) == 1
    assert len(model_output.shape) == 2

    if class_names is not None:
        assert len(class_names) == model_output.shape[1]

    num_classes = model_output.shape[1]

    if class_of_interest is None:
        n_of_rows = math.ceil(num_classes ** 0.5)
        n_of_cols = math.ceil(num_classes / n_of_rows)

        fig, ax = plt.subplots(n_of_rows, n_of_cols, figsize=(3 * n_of_cols, 3 * n_of_rows))

        for class_num in range(num_classes):
            row_num = class_num // n_of_cols
            col_num = class_num % n_of_cols

            precision, recall, thresholds = precision_recall_curve(F.one_hot(targets)[:, class_num],
                                                                   model_output[:, class_num])

            ax[row_num, col_num].plot(recall, precision)

            if class_names is not None:
                subplot_title = class_names[class_num]
            else:
                subplot_title = f"Class {class_num}"
            ax[row_num, col_num].set_title(subplot_title)

            ax[row_num, col_num].set_ylabel('Precision')
            ax[row_num, col_num].set_xlabel('Recall')

        # Hide unused subplots
        for i in range(num_classes, n_of_rows * n_of_cols):
            row_num = i // n_of_cols
            col_num = i % n_of_cols
            fig.delaxes(ax[row_num, col_num])
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

        class_num = class_names.index(class_of_interest)

        precision, recall, thresholds = precision_recall_curve(F.one_hot(targets)[:, class_num],
                                                               model_output[:, class_num])

        ax.plot(recall, precision)

        if class_names is not None:
            subplot_title = class_names[class_num]
        else:
            subplot_title = f"Class {class_num}"

        ax.set_title(subplot_title)

        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()
