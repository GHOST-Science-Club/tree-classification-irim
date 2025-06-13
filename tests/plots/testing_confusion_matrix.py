from visualization_functions import get_confusion_matrix

from pathlib import Path

import torch
import yaml

with open("src/config.yaml", "r") as c:
    config = yaml.safe_load(c)

class_names = config["dataset"]["species_folders"].keys()

num_classes = len(class_names)
print(num_classes)

num_samples = 10000
model_output = torch.rand((num_samples, num_classes))
targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.int64)


get_confusion_matrix(model_output=model_output, targets=targets, filepath=Path.cwd() / "example_plots", class_names=class_names, title="Sample Confusion Matrix", show=True)
