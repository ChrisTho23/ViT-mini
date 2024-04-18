import torch
from torch import nn
from torch.nn import functional as F

class Patch(nn.Module):
    def __init__(self, patch_size: int, latent_space_dim: int):
        super(Patch, self).__init__()
        self.patch_size = patch_size
        self.latent_space_dim = latent_space_dim
        self.fc1 = None # placeholder

    def forward(self, x):
        B, C, H, W = x.shape

        num_patches = (H // self.patch_size) * (W // self.patch_size)

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, num_patches, self.patch_size * self.patch_size)

        patches = patches.view(B, num_patches, -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(C * self.patch_size * self.patch_size, self.latent_space_dim)

        return self.fc1(patches)