import os
from pathlib import Path

import kornia.augmentation as kaug
import torch
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset import ForestDataModule, ForestDataset, OversampledDataset, UndersampledDataset, CurriculumLearningDataset
from callbacks import PrintMetricsCallback, CurriculumLearningCallback
from models.classifier_module import ClassifierModule
from dataset_functions import download_data, load_dataset
from git_functions import get_git_branch, generate_short_hash
from counting_functions import calculate_metrics_per_class, count_metrics
from visualization_functions import (show_n_samples, plot_metrics,
                                     get_confusion_matrix,
                                     get_precision_recall_curve,
                                     get_roc_auc_curve)

import torchvision
import math


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
    freeze = config["training"]["freeze"]
    class_weights = config["training"]["class_weights"] if "class_weights" in config["training"] else None

    if "class_weights" in config["training"] and ("oversample" in config["training"] or "undersample" in config["training"]):
        raise Exception(
            "Can't use class weights and resampling at the same time.")
    weight_decay = config["training"]["weight_decay"]
    model_name = config["model"]["name"]
    image_size = 299 if model_name == "inception_v3" else 224
    transforms = kaug.Resize(size=(image_size, image_size))

    dataset_module = ForestDataset
    dataset_args = {}

    if "oversample" in config["training"]:
        dataset_module = OversampledDataset
        dataset_args = {
            "minority_transform": torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomAffine(
                    degrees=30, translate=(0.1, 0.1), scale=(1, 1.2), shear=10),
            ]),
            "oversample_factor": config["training"]["oversample"]["oversample_factor"],
            "oversample_threshold": config["training"]["oversample"]["oversample_threshold"]
        }
    elif "undersample" in config["training"]:
        dataset_module = UndersampledDataset
        dataset_args = {
            "target_size": config["training"]["undersample"]["target_size"]
        }
    elif "curriculum_learning" in config["training"]:
        dataset_module = CurriculumLearningDataset
        dataset_args = {
            # The list cannot be empty, since the dataloder doesn't accept empty dataset
            "indices": [0]
        }

    datamodule = ForestDataModule(
        dataset['train'],
        dataset['val'],
        dataset['test'],
        dataset=dataset_module,
        dataset_args=dataset_args,
        batch_size=batch_size
    )

    model = ClassifierModule(
        model_name=model_name,
        num_classes=num_classes,
        step_size=config["training"]["step_size"],
        gamma=config["training"]["gamma"],
        freeze=freeze,
        transform=transforms,
        weight=torch.tensor(
            class_weights, dtype=torch.float) if class_weights is not None else None,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # ====================================== TRAINING ========================================== #
    max_epochs = config["training"]["max_epochs"]
    device = config["device"] if torch.cuda.is_available() else "cpu"
    callbacks = [PrintMetricsCallback()]

    if config["training"]["early_stopping"]['apply']:
        callbacks.append(EarlyStopping(monitor=config["training"]["early_stopping"]['monitor'],
                                       patience=config["training"]["early_stopping"]['patience'],
                                       mode=config["training"]["early_stopping"]['mode']))
        checkpoint_dir = config["training"].get(
            "checkpoint_dir", "checkpoints/")
        callbacks.append(ModelCheckpoint(monitor='val_loss',
                                         mode='min',
                                         save_top_k=1,
                                         save_last=False,
                                         dirpath=checkpoint_dir))

    if "curriculum_learning" in config["training"]:
        initial_ratio = config["training"]["curriculum_learning"]["initial_ratio"]
        step_size = config["training"]["curriculum_learning"]["step_size"]
        class_order = config["training"]["curriculum_learning"]["class_order"]
        labels = dataset["train"]["labels"]

        callbacks.append(CurriculumLearningCallback(
            initial_ratio,
            step_size,
            class_order,
            labels
        ))

        min_epochs = math.ceil(num_classes / initial_ratio) * step_size
    else:
        min_epochs = None
        step_size = 0

    branch_name = get_git_branch()
    short_hash = generate_short_hash()
    run_name = f'{branch_name}-{short_hash}'

    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb_run = wandb.init(project="ghost-irim", name=run_name)

    # Log config.yaml to wandb
    wandb.save("src/config.yaml")

    wandb_logger = WandbLogger(
        name=run_name,
        project='ghost-irim',
        log_model=True
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    trainer = Trainer(
        logger=wandb_logger,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=step_size
    )

    trainer.fit(model, datamodule)

    # ====================================== TESTING ========================================== #
    # Retrieve the best checkpoint path from the ModelCheckpoint callback
    best_ckpt_path = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_ckpt_path = callback.best_model_path
            break

    if not best_ckpt_path:
        raise ValueError(
            "No ModelCheckpoint callback found or no best checkpoint available.")

    trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)
    # Callbacks' service
    for callback in callbacks:
        if isinstance(callback, PrintMetricsCallback):
            train_metrics = callback.train_metrics
            val_metrics = callback.val_metrics
            plot_metrics(train_metrics, val_metrics)
            wandb.log({'Accuracy and Loss Curves': wandb.Image(
                'src/plots/acc_loss_curves.png')})

    # Logging plots
    preds = model.predictions
    targets = model.targets

    # Log metrics
    metrics_per_experiment = count_metrics(targets, preds)
    for key, value in metrics_per_experiment.items():
        wandb.log({key: value})

    # Log metrics per class and classnames
    metrics_per_class = calculate_metrics_per_class(targets, preds)
    accs = [metrics_per_class[key]['accuracy']
            for key in metrics_per_class.keys()]
    precs = [metrics_per_class[key]['precision']
             for key in metrics_per_class.keys()]
    recs = [metrics_per_class[key]['recall']
            for key in metrics_per_class.keys()]
    f1s = [metrics_per_class[key]['f1'] for key in metrics_per_class.keys()]
    ious = [metrics_per_class[key]['IoU'] for key in metrics_per_class.keys()]
    names_and_labels = [[key, value] for key, value in label_map.items()]
    logged_metrics = [[name, label, acc, prec, rec, f1, iou] for [
        name, label], acc, prec, rec, f1, iou in zip(names_and_labels, accs, precs, recs, f1s, ious)]

    training_table = wandb.Table(columns=[
                                 'Class name', 'Label', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'IoU'], data=logged_metrics)
    wandb.log({'Classes': training_table})

    # Log confusion matrix, precision-recall curve and roc-auc curve
    get_confusion_matrix(preds, targets, class_names=list(label_map.keys()))
    get_roc_auc_curve(preds, targets, class_names=list(label_map.keys()))
    get_precision_recall_curve(
        preds, targets, class_names=list(label_map.keys()))

    filenames = ['confusion_matrix.png',
                 'precision_recall_curve.png', 'roc_auc_curve.png']
    titles = ['Confusion Matrix', 'Precision-Recall Curve', 'ROC AUC Curve']
    for filename, title in zip(filenames, titles):
        wandb.log({title: wandb.Image(f'src/plots/{filename}')})

    onnx_filepath = "tile_processing_model.onnx"
    model.to_onnx(
        onnx_filepath,
        export_params=True,
        input_names=["input"],   # Name of the input tensor
        output_names=["output"],  # Name of the output tensor
        dynamic_axes={
            'input': {0: 'batch_size'},  # Make batch size dynamic
            'output': {0: 'batch_size'}
        }
    )
    wandb.artifact = wandb.Artifact(
        name=f"onnx-model-{wandb_run.id}",
        type="model",
        description=f"ONNX model from run {wandb_run.id}",
        metadata=dict(wandb_run.config)
    )
    wandb.artifact.add_file(onnx_filepath)

    wandb.finish()


if __name__ == "__main__":
    main()
