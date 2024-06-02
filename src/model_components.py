from torch import nn
import torch

from src.attention import MultiHeadSelfAttentionLayer

class FeedForward(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, dim_out: int , dtype=float):
        super(FeedForward, self).__init__()
        self.dtype=dtype
        self.ff_1 = nn.Linear(dim_in, dim_ff, bias=True, dtype=self.dtype)
        self.activation = nn.GELU()
        self.ff_2 = nn.Linear(dim_ff, dim_out, bias=True, dtype=self.dtype)
    def forward(self, x):
        return self.ff_2(self.activation(self.ff_1(x)))


class ClassificationHead(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, num_classes: int , dtype=float):
        super(ClassificationHead, self).__init__()
        self.dtype=dtype
        self.ff_1 = nn.Linear(dim_in, dim_ff, bias=True, dtype=self.dtype)
        self.activation = nn.Tanh() # tanh activation
        self.ff_2 = nn.Linear(dim_ff, num_classes, bias=True, dtype=self.dtype)

    def forward(self, x):
        return self.ff_2(self.activation(self.ff_1(x))) # B, N, num_classes
    

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, dim_ff: int, num_heads: int, context_length: int, dtype=float):
        super(TransformerBlock, self).__init__()
        self.dtype=dtype
        self.layer_norm_1 = nn.LayerNorm(emb_dim, dtype=self.dtype) # only normalize over embedding dim.
        self.layer_norm_2 = nn.LayerNorm(emb_dim, dtype=self.dtype)
        self.multi_head_att = MultiHeadSelfAttentionLayer(
            dim_in=emb_dim, dim_out=emb_dim, num_heads=num_heads, context_length=context_length, dtype=self.dtype
        )
        self.feed_forward = FeedForward(dim_in=emb_dim, dim_ff=dim_ff, dim_out=emb_dim, dtype=self.dtype)
    def forward(self, x):
        x_prime = x + self.multi_head_att(self.layer_norm_1(x))
        return x_prime+self.feed_forward(self.layer_norm_2(x_prime))
