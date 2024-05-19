import torch
from torch import nn
import torch.nn.functional as F

from src.model_components import ClassificationHead, TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(
        self, image_width: int, image_height: int, channel_size: int,
        patch_size: int, latent_space_dim: int, dim_ff: int,  num_heads: int,
        depth: int, num_classes: int = None
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
            self.num_patches+1, # effective sequence length is num_patches + 1 (class token)
            self.latent_space_dim
        )
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(latent_space_dim, dim_ff, num_heads, self.num_patches) for _ in range(depth)]
        )
        self.classification_head = ClassificationHead(latent_space_dim, dim_ff, num_classes) # only one token used

    def forward(self, x, targets=None):
        B, C, H, W = x.shape

        # extract patches
        x = x.unfold(
            2, self.patch_size, self.patch_size
        ).unfold(3, self.patch_size, self.patch_size) # B, C, H//patch_size, W//patch_size, patch_size, patch_size
        x = x.contiguous().view(
            B, self.num_patches, C * self.patch_size * self.patch_size
        ) # B, num_patches, C * patch_size * patch_size

        # class token
        x_class = torch.zeros(B, 1, self.latent_space_dim, dtype=float)

        # create embeddings
        x_patch_embedding = self.patch_embedding_layer(x) # B, num_patches, latent_space_dim
        x_pos_embedding = self.positional_embedding_table(
            torch.arange(self.num_patches+1, device=x.device)
        ) # B, num_patches, latent_space_dim
        x_patch_embedding = torch.cat((x_class, x_patch_embedding), dim=1) # concatenate along sequence dim. 
        x_embedding = x_patch_embedding + x_pos_embedding # B, num_patches, latent_space_dim

        # transformer blocks
        x = self.transformer_blocks(x_embedding)# .view(B, -1) # B, num_patches * latent_space_dim

        # classification head
        logits = self.classification_head(x[:, 0, :]) # only feed transformed class token into classification head
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss