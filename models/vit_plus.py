import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PatchEmbedding(nn.Module):
    """
    Split image into patches and project them into an embedding space.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class MultiScaleAttention(nn.Module):
    """
    Advanced multi-scale self-attention for hybrid feature extraction.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):