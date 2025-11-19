import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image

from models.classifier_module import ClassifierModule
from models.segmentation_wrapper import SegmentationWrapper
from dataset_functions import load_dataset
from dataset import ForestDataset
from transforms import Transforms
import onnx
import json


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
    transforms = Transforms(image_size=(image_size, image_size))

    # =========================== DATA LOADING ===================================== #
    dataset_folder = Path.cwd() / config.dataset.folder
    dataset_folder.mkdir(exist_ok=True)

    dataset, label_map = load_dataset(dataset_folder, config.dataset.species_folders)

    test_data = dataset["test"]
    test_dataset = ForestDataset(test_data["paths"], test_data["labels"], transform=transforms)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2
    )

    num_classes = len(label_map)

    # =========================== MODEL LOADING ==================================== #
    wandb_artifact = config.inference.get("wandb_artifact", None)
    
    if wandb_artifact:
        wandb_project = config.inference.get("wandb_project", "ghost-irim")
        checkpoint_path = download_checkpoint_from_wandb(wandb_artifact, wandb_project)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please set 'wandb_artifact' in config.yaml to download from W&B, "
            "or ensure the local checkpoint exists."
        )
    
    print(f"Loading model from: {checkpoint_path}")

    classifier = ClassifierModule.load_from_checkpoint(
        checkpoint_path,
        model_name=model_name,
        num_classes=num_classes,
    )
    classifier = classifier.to(device).eval()

    seg_model = SegmentationWrapper(classifier, mask_size=mask_size).to(device)
    seg_model.eval()

    output_dir = Path("segmentation_outputs")
    output_dir.mkdir(exist_ok=True)

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

        add_meta('model_type', 'Segmentor')
        add_meta('class_names', class_names)
        add_meta('resolution', 20)
        add_meta('tiles_size', image_size)
        add_meta('tiles_overlap', 0)
        add_meta('standardization_mean', [0.5, 0.5, 0.5]) 
        add_meta('standardization_std', [0.5, 0.5, 0.5]) # TODO: make sure these two are correct

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
                }
            )
            onnx_artifact.add_file(str(onnx_path))
            wandb.log_artifact(onnx_artifact)
            print(f"ONNX model uploaded to W&B artifacts as 'segmentation-model-{model_name}'")
        else:
            print("Warning: W&B run not initialized. ONNX model not uploaded to artifacts.")
    
    # =========================== INFERENCE LOOP =================================== #
    print(f"Running inference on {len(test_loader)} samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            imgs, _ = batch
            imgs = imgs.to(device)

            masks = seg_model(imgs)

if __name__ == "__main__":
    main()
