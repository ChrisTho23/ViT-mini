import torch
from torch import nn
import numpy as np

from attention import multi_head_self_attention_layer


class Patch(nn.Module):
    def __init__(self, patch_size: int, latent_space_dim: int):
        super(Patch, self).__init__()
        self.patch_size = patch_size
        self.latent_space_dim = latent_space_dim
        self.fc1 = nn.Linear(self.patch_size * self.patch_size, self.latent_space_dim, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, num_patches, C * self.patch_size * self.patch_size)

class feed_forward(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, dim_out: int , dtype=float):
        super(feed_forward, self).__init__()
        self.dtype=dtype
        self.ff_1 = nn.Linear(dim_in, dim_ff, bias=True, dtype=self.dtype)
        self.ff_2 = nn.Linear(dim_ff, dim_out, bias=True, dtype=self.dtype)
        self.activation = nn.GELU()
    def forward(self, x):
        return self.ff_2(self.activation(self.ff_1(x)))
    

class transformer_block(nn.Module):
    def __init__(self,emb_dim: int , dim_ff: int, num_heads: int, context_length: int, dtype=float):
        super(transformer_block, self).__init__()
        self.dtype=dtype
        self.layer_norm_1 = nn.LayerNorm(emb_dim, dtype=self.dtype) # only normalize over embedding dim.
        self.layer_norm_2 = nn.LayerNorm(emb_dim, dtype=self.dtype)
        self.multi_head_att = multi_head_self_attention_layer(dim_in=emb_dim, dim_out=emb_dim, num_heads=num_heads, context_length=context_length, dtype=self.dtype)
        self.feed_forward = feed_forward(dim_in=emb_dim, dim_ff=dim_ff, dim_out =emb_dim, dtype=self.dtype)
    def forward(self, x):
        x_prime = x + self.multi_head_att(self.layer_norm_1(x))
        return x_prime+self.feed_forward(self.layer_norm_2(x_prime))


if(__name__=="__main__"):
    # test position-wise feed forward
    test_input = torch.tensor(np.random.rand(1,10,8)) # B: 1, N: 10, D: 5
    feedforw = feed_forward(dim_in=8, dim_ff=10, dim_out=8)
    test_output = feedforw(test_input) 
    print("Test output: ", test_output)

    # test transformer block
    transf = transformer_block(emb_dim=8, dim_ff=10, num_heads=2, context_length=10)
    test_transf_output = transf(test_input)
    print("Test transformer output: ", test_transf_output)
