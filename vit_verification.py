"""This script is used to verify the correctness of the vit_verification.py script.
"""

import numpy as np
import timm
import torch
from torchinfo import summary
from vit_scratch import VisionTransformer


# helper function
def get_n_params(module):
    """Count the number of parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_tensor_equal(tensor1, tensor2):
    """Assert that two tensors are equal."""
    np.testing.assert_allclose(
        tensor1.detach().cpu().numpy(), tensor2.detach().cpu().numpy()
    )


# Load the model
MODEL_NAME = "vit_base_patch16_224"
MODEL_OFFICIAL = timm.create_model(MODEL_NAME, pretrained=True)

print(type(MODEL_OFFICIAL))


print(summary(MODEL_OFFICIAL, input_size=(2, 3, 224, 224)))
CUSTOM_CONFIG = {
    "img_size": 224,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4,
}

MODEL_CUSTOM = VisionTransformer(**CUSTOM_CONFIG).cuda()
MODEL_CUSTOM.eval()

for (n_o, p_o), (n_c, p_c) in zip(
    MODEL_OFFICIAL.named_parameters(), MODEL_CUSTOM.named_parameters()
):
    assert p_o.numel() == p_c.numel()

    print(f"{n_o}| {n_c}")

    p_c.data[:] = p_o.data

    assert_tensor_equal(p_c.data, p_o.data)


INP = torch.randn(1, 3, 224, 224).cuda()
RES_C = MODEL_CUSTOM(INP)
RES_O = MODEL_OFFICIAL(INP)

# Assert
assert get_n_params(MODEL_CUSTOM) == get_n_params(MODEL_OFFICIAL)
assert_tensor_equal(RES_C, RES_O)

# save the model
torch.save(MODEL_CUSTOM, "vit_verification.pth")
