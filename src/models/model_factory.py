from torchvision import models as tv_models
from transformers import ViTForImageClassification
import torch.nn as nn


def create_model(model_name, num_classes, freeze=False):
    if "resnet" in model_name:
        models = {
            "resnet18": tv_models.resnet18(weights="DEFAULT"),
            "resnet34": tv_models.resnet34(weights="DEFAULT"),
            "resnet50": tv_models.resnet50(weights="DEFAULT"),
            "resnet101": tv_models.resnet101(weights="DEFAULT"),
            "resnet152": tv_models.resnet152(weights="DEFAULT"),
        }
        try:
            model = models[model_name]
        except KeyError:
            print(f"Model '{model_name}' not supported, pick one of {models.keys()}. Using resnet18.")
            model = models["resnet18"]
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "inception_v3":
        model = tv_models.inception_v3(weights="DEFAULT", aux_logits=True)
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

    elif "efficientnet" in model_name:
        models = {
            "efficientnet-b0": tv_models.efficientnet_b0(weights="DEFAULT"),
            "efficientnet-b1": tv_models.efficientnet_b1(weights="DEFAULT"),
            "efficientnet-b2": tv_models.efficientnet_b2(weights="DEFAULT"),
            "efficientnet-b3": tv_models.efficientnet_b3(weights="DEFAULT"),
            "efficientnet-b4": tv_models.efficientnet_b4(weights="DEFAULT"),
            "efficientnet-b5": tv_models.efficientnet_b5(weights="DEFAULT"),
            "efficientnet-b6": tv_models.efficientnet_b6(weights="DEFAULT"),
            "efficientnet-b7": tv_models.efficientnet_b7(weights="DEFAULT"),
            "efficientnet-v2-s": tv_models.efficientnet_v2_s(weights="DEFAULT"),
            "efficientnet-v2-m": tv_models.efficientnet_v2_m(weights="DEFAULT"),
            "efficientnet-v2-l": tv_models.efficientnet_v2_l(weights="DEFAULT"),
        }
        try:
            model = models[model_name]
        except KeyError:
            print(f"Model '{model_name}' not supported, pick one of the following: {models.keys()}. Using efficientnet-b0 as default.")
            model = tv_models.efficientnet_b0(weights="DEFAULT")

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return model
