import torch
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(
        self, image_width: int, image_height: int, channel_size: int,
        patch_size: int, latent_space_dim: int
    ):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.latent_space_dim = latent_space_dim
        self.num_patches = (image_width * image_height) // (self.patch_size**2)
        self.patch_embedding_layer = nn.Linear(
            channel_size * self.patch_size * self.patch_size,  
            self.latent_space_dim  
        )
        self.positional_embedding_table = nn.Embedding(
            self.num_patches,
            self.latent_space_dim
        )

    def forward(self, x):
        B, C, H, W = x.shape

        patches = x.unfold(
            2, self.patch_size, self.patch_size
        ).unfold(3, self.patch_size, self.patch_size) # B, C, H//patch_size, W//patch_size, patch_size, patch_size
        patches = patches.contiguous().view(
            B, self.num_patches, C * self.patch_size * self.patch_size
        ) # B, num_patches, C * patch_size * patch_size

        patch_embedding = self.patch_embedding_layer(patches) # B, num_patches, latent_space_dim
        positional_embedding = self.positional_embedding_table(
            torch.arange(self.num_patches, device=patches.device)
        ) # B, num_patches, latent_space_dim
        embedding = patch_embedding + positional_embedding # B, num_patches, latent_space_dim

        return embedding