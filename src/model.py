import torch
from torch import nn
from torch.nn import functional as F

from model_components import Patch

class VisionTransformer(nn.Module):
    def __init__(self, image_width: int, image_height: int, patch_size: int, latent_space_dim: int, num_classes: int):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.latent_space_dim = latent_space_dim
        self.num_patches = (image_width * image_height) // (self.patch_size * self.patch_size)
        self.num_classes = num_classes
        self.pixel_embedding_table = nn.Embedding(
            self.patch_size * self.patch_size, 
            self.latent_space_dim
        )
        self.positional_embedding_table = nn.Parameter(
            self.num_patches,
            self.latent_space_dim
        )

    def forward(self, x):
        B, C, H, W = x.shape

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, self.num_patches, C * self.patch_size * self.patch_size)

        pixel_embedding = self.pixel_embedding_table(patches)
        positional_embedding = self.positional_embedding_table(
            torch.arange(self.num_patches, device=patches.device)
        )

        return patches