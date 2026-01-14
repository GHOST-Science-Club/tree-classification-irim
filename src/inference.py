import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
from PIL import Image

from models.classifier_module import ClassifierModule
from models.segmentation_wrapper import SegmentationWrapper
from dataset_functions import load_dataset
from dataset import ForestDataset
# from transforms import Transforms
from counting_functions import calculate_metrics_per_class, count_metrics
from visualization_functions import get_confusion_matrix, get_precision_recall_curve, get_roc_auc_curve
import onnx
import json
import kornia.augmentation as kaug
from kornia import image_to_tensor
import torch.nn as nn
from torch.utils.data import Dataset


class InferenceTransform(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.resize = kaug.Resize(size=size, keepdim=True)

    @torch.no_grad()
    def forward(self, x):
        # x is numpy array (H, W, C)
        x_t = image_to_tensor(x, keepdim=True).float()
        x_res = self.resize(x_t)
        return x_res.squeeze(0)


class RawImageDataset(Dataset):
    """Dataset for loading raw images without labels (for ONNX-style inference testing)"""
    def __init__(self, image_dir, transform=None, image_extensions=('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Get all image files
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(sorted(self.image_dir.glob(f"*{ext}")))
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image as numpy array (similar to ONNX input)
        with Image.open(image_path) as img:
            if img.mode == "RGBA":
                image = img.convert("RGB")
            else:
                image = img.copy()
            
            # Convert to numpy array [H, W, C] in [0, 255] range
            image_np = np.array(image)
        
        # Apply transform if provided
        if self.transform:
            image_tensor = self.transform(image_np)
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        return image_tensor, str(image_path.name)


def download_checkpoint_from_wandb(artifact_path, project_name="ghost-irim"):
    print(f"Downloading checkpoint from W&B: {artifact_path}")

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    run = wandb.init(project=project_name, job_type="inference")

    artifact = run.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download()

    artifact_path_obj = Path(artifact_dir)
    checkpoint_files = list(artifact_path_obj.glob("*.ckpt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt file found in artifact directory: {artifact_dir}")

    checkpoint_path = checkpoint_files[0]
    print(f"Checkpoint downloaded to: {checkpoint_path}")

    return checkpoint_path


def main():
    # =========================== CONFIG & SETUP ================================== #
    config = OmegaConf.load("src/config.yaml")

    config_device = config.device
    if config_device in ["gpu", "cuda"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model_name = config.model.name
    mask_size = config.inference.get("mask_size", 224)
    image_size = 299 if model_name == "inception_v3" else 224
    transforms = InferenceTransform(size=(image_size, image_size))

    # =========================== DATA LOADING ===================================== #
    use_raw_images = config.inference.get("use_raw_images", False)
    
    if use_raw_images:
        # Load raw images from specified directory
        raw_images_path = config.inference.get("raw_images_path", "test_images_qgis/10122025_193353")
        raw_images_dir = Path.cwd() / raw_images_path
        
        if not raw_images_dir.exists():
            raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")
        
        print(f"\n{'='*60}")
        print(f"Running inference on RAW IMAGES (ONNX-style testing)")
        print(f"Directory: {raw_images_dir}")
        print(f"{'='*60}\n")
        
        test_dataset = RawImageDataset(raw_images_dir, transform=transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # Load label map for model initialization (but no ground truth labels available)
        dataset_folder = Path.cwd() / config.dataset.folder
        dataset_folder.mkdir(exist_ok=True)
        dataset, label_map = load_dataset(dataset_folder, config.dataset.species_folders)
        num_classes = len(label_map)
        has_labels = False
    else:
        # Load test dataset with labels
        dataset_folder = Path.cwd() / config.dataset.folder
        dataset_folder.mkdir(exist_ok=True)

        dataset, label_map = load_dataset(dataset_folder, config.dataset.species_folders)

        test_data = dataset["test"]
        test_dataset = ForestDataset(test_data["paths"], test_data["labels"], transform=transforms)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        num_classes = len(label_map)
        has_labels = True

    # =========================== MODEL LOADING ==================================== #
    wandb_artifact = config.inference.get("wandb_artifact", None)

    if wandb_artifact:
        wandb_project = config.inference.get("wandb_project", "ghost-irim")
        checkpoint_path = download_checkpoint_from_wandb(wandb_artifact, wandb_project)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please set 'wandb_artifact' in config.yaml to download from W&B, or ensure the local checkpoint exists.")

    print(f"Loading model from: {checkpoint_path}")

    classifier = ClassifierModule.load_from_checkpoint(
        checkpoint_path,
        model_name=model_name,
        num_classes=num_classes,
    )
    classifier = classifier.to(device).eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    seg_model = SegmentationWrapper(
        classifier,
        mask_size=mask_size,
        mean=mean,  # TODO: fix
        std=std,  # TODO: fix
        input_rescale=True,  # Expects 0-255 input, scales to 0-1 internally
    ).to(device)
    seg_model.eval()

    # =========================== EXPORT TO ONNX =================================== #
    if config.inference.get("export_onnx", False):
        dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
        onnx_path = Path("segmentation_model.onnx")
        torch.onnx.export(
            seg_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["mask"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch_size"}, "mask": {0: "batch_size"}},
            do_constant_folding=True,
        )
        print(f"Exported model to {onnx_path.resolve()}")

        # Add metadata
        model_onnx = onnx.load(onnx_path)

        class_names = {v: k for k, v in label_map.items()}

        def add_meta(key, value):
            meta = model_onnx.metadata_props.add()
            meta.key = key
            meta.value = json.dumps(value)

        add_meta("model_type", "Segmentor")
        add_meta("class_names", class_names)
        add_meta("resolution", 20)
        add_meta("tiles_size", image_size)
        add_meta("tiles_overlap", 0)

        onnx.save(model_onnx, onnx_path)

        if wandb.run is not None:
            onnx_artifact = wandb.Artifact(
                name=f"segmentation-model-{model_name}",
                type="model",
                description=f"ONNX segmentation model ({model_name}, {num_classes} classes)",
                metadata={
                    "model_name": model_name,
                    "num_classes": num_classes,
                    "image_size": image_size,
                    "format": "onnx",
                    "opset_version": 17,
                },
            )
            onnx_artifact.add_file(str(onnx_path))
            wandb.log_artifact(onnx_artifact)
            print(f"ONNX model uploaded to W&B artifacts as 'segmentation-model-{model_name}'")
        else:
            print("Warning: W&B run not initialized. ONNX model not uploaded to artifacts.")

    # =========================== INFERENCE LOOP =================================== #
    print(f"Running inference on {len(test_loader)} samples...")
    all_preds = []
    all_targets = []
    all_filenames = []

    # Debug: check first batch input values
    debug_first_batch = True

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if has_labels:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                all_targets.append(labels)
            else:
                imgs, filenames = batch
                imgs = imgs.to(device)
                all_filenames.extend(filenames)

            # Debug: print input statistics for first batch
            if debug_first_batch:
                print(f"\n[DEBUG] Input tensor stats:")
                print(f"  Shape: {imgs.shape}")
                print(f"  dtype: {imgs.dtype}")
                print(f"  Min: {imgs.min().item():.4f}, Max: {imgs.max().item():.4f}")
                print(f"  Mean: {imgs.mean().item():.4f}, Std: {imgs.std().item():.4f}")
                
                # Check what SegmentationWrapper receives after rescale
                if seg_model.input_rescale:
                    rescaled = imgs / 255.0
                    print(f"\n[DEBUG] After /255 rescale:")
                    print(f"  Min: {rescaled.min().item():.4f}, Max: {rescaled.max().item():.4f}")
                    
                    normalized = (rescaled - seg_model.mean) / seg_model.std
                    print(f"\n[DEBUG] After normalization:")
                    print(f"  Min: {normalized.min().item():.4f}, Max: {normalized.max().item():.4f}")
                
                debug_first_batch = False

            masks = seg_model(imgs)

            probs = masks[:, :, 0, 0]

            all_preds.append(probs)

    all_preds = torch.cat(all_preds, dim=0)
    if has_labels:
        all_targets = torch.cat(all_targets, dim=0)

    # =========================== METRICS & LOGGING ================================ #
    if has_labels:
        # Calculate and log metrics only when we have ground truth labels
        if wandb.run is not None:
            print("Calculating and logging metrics...")

            metrics_per_experiment = count_metrics(all_targets, all_preds)
            print(f"Test Metrics: {metrics_per_experiment}")
            for key, value in metrics_per_experiment.items():
                wandb.log({key: value})

            metrics_per_class = calculate_metrics_per_class(all_targets, all_preds)
            accs = [metrics_per_class[key]["accuracy"] for key in metrics_per_class.keys()]
            precs = [metrics_per_class[key]["precision"] for key in metrics_per_class.keys()]
            recs = [metrics_per_class[key]["recall"] for key in metrics_per_class.keys()]
            f1s = [metrics_per_class[key]["f1"] for key in metrics_per_class.keys()]
            ious = [metrics_per_class[key]["IoU"] for key in metrics_per_class.keys()]
            names_and_labels = [[key, value] for key, value in label_map.items()]
            logged_metrics = [[name, label, acc, prec, rec, f1, iou] for [name, label], acc, prec, rec, f1, iou in zip(names_and_labels, accs, precs, recs, f1s, ious, strict=False)]

            training_table = wandb.Table(columns=["Class name", "Label", "Accuracy", "Precision", "Recall", "F1-score", "IoU"], data=logged_metrics)
            wandb.log({"Classes": training_table})

            plots_dir = Path("src/plots")
            plots_dir.mkdir(exist_ok=True, parents=True)

            get_confusion_matrix(all_preds, all_targets, class_names=list(label_map.keys()))
            get_roc_auc_curve(all_preds, all_targets, class_names=list(label_map.keys()))
            get_precision_recall_curve(all_preds, all_targets, class_names=list(label_map.keys()))

            filenames = ["confusion_matrix.png", "precision_recall_curve.png", "roc_auc_curve.png"]
            titles = ["Confusion Matrix", "Precision-Recall Curve", "ROC AUC Curve"]
            for filename, title in zip(filenames, titles, strict=False):
                wandb.log({title: wandb.Image(f"src/plots/{filename}")})
        else:
            print("W&B run not active. Skipping metrics logging.")
    else:
        # For raw images, just output predictions
        print("\n" + "="*60)
        print("RAW IMAGE INFERENCE RESULTS")
        print("="*60)
        
        # Get predicted classes
        pred_classes = torch.argmax(all_preds, dim=1)
        
        # Create reverse label map
        reverse_label_map = {v: k for k, v in label_map.items()}
        
        # Save predictions to file
        output_dir = Path("src/plots/raw_predictions")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results_file = output_dir / "predictions.txt"
        with open(results_file, "w") as f:
            f.write("Filename\tPredicted_Class\tPredicted_Label\tConfidences\n")
            for i, (filename, pred_class, probs) in enumerate(zip(all_filenames, pred_classes, all_preds)):
                pred_label = reverse_label_map[pred_class.item()]
                confidence_str = "\t".join([f"{reverse_label_map[j]}:{probs[j].item():.4f}" for j in range(num_classes)])
                f.write(f"{filename}\t{pred_class.item()}\t{pred_label}\t{confidence_str}\n")
                
                # Print first 10 predictions
                if i < 10:
                    print(f"{filename}: {pred_label} (confidence: {probs[pred_class].item():.4f})")
        
        print(f"\n...")
        print(f"\nFull predictions saved to: {results_file}")
        print(f"Total images processed: {len(all_filenames)}")
        print("="*60)


if __name__ == "__main__":
    main()
