from torchvision import models as tv_models
from transformers import ViTForImageClassification
import torch.nn as nn


def create_model(model_name, num_classes, freeze=False):
    if model_name == "resnet18":
        model = tv_models.resnet18(weights="DEFAULT")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "inception_v3":
        model = tv_models.inception_v3(weights="DEFAULT", aux_logits=False)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes
        )
        if freeze:
            for param in model.vit.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    
    return model
