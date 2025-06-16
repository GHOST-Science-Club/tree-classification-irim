import random
import pytest
from PIL import Image


@pytest.fixture
def sample_data(tmp_path):
    """Creates data instance with ten sample images."""
    num_images = 10
    data = {"paths": [], "labels": []}

    # Define specific labels for the first two images
    predefined_labels = {0: 0, 1: 1}

    for i in range(num_images):
        img_path = tmp_path / f"sample_image{i}.jpg"

        img = Image.new("RGB", (224, 224), color=tuple(random.randint(0, 255) for _ in range(3)))
        img.save(img_path)

        # Assign predefined labels for the first two images,
        # random for the rest
        label = predefined_labels.get(i, random.randint(0, 9))

        data["paths"].append(img_path)
        data["labels"].append(label)

    return data
