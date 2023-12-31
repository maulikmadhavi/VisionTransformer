"""This script is used to verify the correctness of the vit_verification.py script.
"""

import json
import urllib.request

import numpy as np
import timm
import torch
from PIL import Image
from torchinfo import summary
from torchvision import transforms
from vit_scratch import VisionTransformer

# helper function
def get_n_params(module):
    """Count the number of parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_tensor_equal(tensor1, tensor2):
    """Assert that two tensors are equal."""
    np.testing.assert_allclose(tensor1.detach().cpu().numpy(), tensor2.detach().cpu().numpy())


# Load the model


MODEL_NAME = "vit_base_patch16_224.augreg_in21k"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_OFFICIAL = timm.create_model(MODEL_NAME, pretrained=True)

print(type(MODEL_OFFICIAL))

# get model specific transforms (normalization, resize)
DATA_CONFIG = timm.data.resolve_model_data_config(MODEL_OFFICIAL)
TRANSFORMS = timm.data.create_transform(**DATA_CONFIG, is_training=False)


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
    "n_classes": 1000,
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
# save the model
# -->torch.save(MODEL_CUSTOM, "vit_verification.pth")


RAW_IMG = Image.open("cat.jpg")
IMG = TRANSFORMS(RAW_IMG).unsqueeze(0).cuda()
print(IMG.shape)
print(TRANSFORMS)

TRANSFORMS2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
MODEL_CUSTOM.eval()
MODEL_OFFICIAL.eval()

# open the URL and read the response
with urllib.request.urlopen(
    "https://gist.githubusercontent.com/marodev/7b3ac5f63b0fc5ace84fa723e72e956d/raw/2c48d37df5c93a2370cecb7415744b01a6154c47/imagenet.json"
) as response:
    DATA = json.load(response)
 # parse the JSON data

DATA = {int(k): v for k, v in DATA.items()}


# Load Imagenet1k class names
print("Testing the image using Timm's model and transforms")
with torch.no_grad():
    out = MODEL_OFFICIAL(IMG).softmax(dim=1)
    v, ind = out.topk(5, dim=1)

    v = (100.0 * v).flatten().detach().tolist()
    ind = ind.flatten().detach().tolist()
    for ix, (v_, ind_) in enumerate(zip(v, ind)):
        print(f"Rank:{ix},  Confidence: {v_:0.4f},  Id: {ind_},  Name: {DATA[ind_]} ")


IMG = TRANSFORMS2(RAW_IMG).unsqueeze(0).cuda()
print("Testing the image using Timm's model and custom transforms")
with torch.no_grad():
    out = MODEL_OFFICIAL(IMG).softmax(dim=1)
    v, ind = out.topk(5, dim=1)

    v = (100.0 * v).flatten().detach().tolist()
    ind = ind.flatten().detach().tolist()
    for ix, (v_, ind_) in enumerate(zip(v, ind)):
        print(f"Rank:{ix},  Confidence: {v_:0.4f},  Id: {ind_},  Name: {DATA[ind_]} ")


print("Testing the image using custom model and custom transforms")
with torch.no_grad():
    out = MODEL_CUSTOM(IMG).softmax(dim=1)
    v, ind = out.topk(5, dim=1)
    v = (100.0 * v).flatten().detach().tolist()
    ind = ind.flatten().detach().tolist()
    for ix, (v_, ind_) in enumerate(zip(v, ind)):
        print(f"Rank:{ix},  Confidence: {v_:0.4f},  Id: {ind_},  Name: {DATA[ind_]} ")