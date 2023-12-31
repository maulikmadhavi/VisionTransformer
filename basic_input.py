# Vision transfomer:
# An Image is Worth 16 x 16 Grid
# https://www.youtube.com/watch?v=_mFHxLRrTd4
# vision transformer and Deit using PyTorch Lightning

# 1. Image is divided into 16 x 16 Grid square patch
# 2. Each patch is flattened into 1D array
# 3. To allow the model to learn about the structure of image add learnable positional embedding to each patch


import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(10, 3, 256, 256)
# b = 10 batch size, 3 channels, 256 height, 256 width
# x = torch.randn(bs, nc, h, w)
# We may divide the height and width by 16 to get the number of patches
# we call the new height and width h1 and w1
# h1 = h / 16 => h = h1 * 16
# w1 = w / 16 => w = w1 * 16
# Let height division factor says pt1 and width division factor says pt2
# h = h1 * pt1 and w = w1 * pt2 (pt1 and pt2 are 16)

pt1 = 16
pt2 = 16
c = 3
x = einops.rearrange(x, "b c (h1 pt1) (w1 pt2) -> b h1 w1 (pt1 pt2 c)", pt1=16, pt2=16)
# pt1 = 16, pt2 = 16, c = 3 => pt1 * pt2 * c = 3 * 16 * 16 = 768
# Output: b h w (pt1 pt2 c) = 10 16 16 768


# Next we transforms the input to the embedding vector space
# Let d_emb = 768 (embedding vector size)
d_emb = 768
l_emb = nn.Linear(pt1 * pt2 * c, d_emb)


# Adding [CLS] token to the input
l_cls = nn.parameter(
    torch.randn(1, 1, d_emb)
)  # 1 x 1 x d_emb  =>  embedding vector for cls token learnable

# Adding positional embedding to the input
num_patches = 16
l_pos = nn.parameter(
    torch.randn((num_patches) ** 2 + 1, d_emb)
)  #  (16 * 16 + 1) * d_emb
# assign each vector to the each patch (total number of patches are 16 * 16 + 1)


# DeiT (Data Effiencet Image transformer and distillation through attention)
# Problem in ViT is the need to be trained on hundreds of millions of images using
# an expensive infrastructure.
# To overcome this problem, distillation through attention is used where teacher
# model is convnet and student model is the transformer.

# The student model pursues two objectives:
# 1. learning from  a labeled dataset (strong supervision)
# 2. learning from the teacher (distillation) that produces distillation token uses as input-output to the DeiT model (student)
