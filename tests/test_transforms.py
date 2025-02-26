import pytest
import torch
import numpy as np
from src.transforms import Preprocess, Transforms
from kornia.utils import image_to_tensor


@pytest.mark.transforms
@pytest.mark.parametrize(
    "input_shape",
    [(3, 64, 64), (1, 128, 128), (3, 32, 32)],  # Different image sizes and channels
)
def test_preprocess_kornia(input_shape):
    """Test the Preprocess module with different input shapes."""

    # Create a random NumPy image (values between 0 and 255)
    np_image = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)

    torch_input = torch.tensor(np_image)

    preprocess = Preprocess()

    output = preprocess(torch_input)

    expected_output = image_to_tensor(np_image, keepdim=True).float() / 255.0

    # Check output shape matches expected shape
    assert output.shape == expected_output.shape, "Output shape mismatch"

    # Check values are normalized between 0 and 1
    assert torch.all((output >= 0) & (output <= 1)), "Output values not normalized correctly"

    # Check dtype is float32
    assert output.dtype == torch.float32, "Output dtype is not float32"
    