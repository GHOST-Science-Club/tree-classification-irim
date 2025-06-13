import pytest
import torch
import numpy as np
from src.transforms import Preprocess
from kornia.utils import image_to_tensor


@pytest.mark.transforms
@pytest.mark.parametrize(
    "input_shape",
    # Different image sizes and channels
    [(3, 64, 64), (1, 128, 128), (3, 32, 32)],
)
def test_preprocess_kornia(input_shape):
    # Create a random NumPy image (values between 0 and 255)
    np_image = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)

    torch_input = torch.tensor(np_image)
    preprocess = Preprocess()
    output = preprocess(torch_input)
    expected_output = image_to_tensor(np_image, keepdim=True).float() / 255.0

    error_msg = {"bad-shape": "Output shape mismatch", "no-norm": "Output values not normalized correctly", "bad-type": "Output dtype is not float32"}

    assert output.shape == expected_output.shape, error_msg["bad-shape"]
    assert torch.all((output >= 0) & (output <= 1)), error_msg["no-norm"]
    assert output.dtype == torch.float32, error_msg["bad-type"]
