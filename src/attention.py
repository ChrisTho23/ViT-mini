import torch

class self_attention_layer(torch.nn.Module):
    def __init__(self,dim_in, dim_out):
        super(self_attention_layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.keys_layer = torch.nn.Linear(dim_in, dim_out, bias = False)
        self.queries_layer = torch.nn.Linear(dim_in, dim_out, bias = False)
        self.values_layer = torch.nn.Linear(dim_in, dim_out, bias=False)
        self.softmax_layer = torch.nn.Softmax(dim=-1)
    def get_att_scores(self, keys, queries):
        """
        Implements computation of attention scores:
        :params keys: keys [B, N, D]
        :params queries: [B, N, D]
        """
        return self.softmax_layer(torch.einsum('bid, bjd->bij', queries, keys))
    def forward(self, x):
        """
        Implements forward pass of self-attention layer
        params:
        x: [B, N, D] (B: batch size, N: context-length, D: latent space dim.)
        """
        B, N, D = x.shape

        x = torch.nn.LayerNorm([N,D])(x)
        keys = self.keys_layer(x)
        queries = self.queries_layer(x)
        values = self.values_layer(x)

        # attention mechanism
        att_scores = self.get_att_scores(keys, queries)

        return torch.einsum('bjd, bij->bid', values, att_scores) # shape: [B, N, D]

class multi_head_self_attention_layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, context_length):
        super(multi_head_self_attention_layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.d_head = dim_out // num_heads
        self.context_length = context_length
        self.output_projection = torch.nn.Linear(dim_out, dim_out)
        self.layer_norm =torch.nn.LayerNorm({self.context_length, self.dim_in})

        self.attention_heads = [self_attention_layer(dim_in, self.d_head) for i in self.num_heads]
    def forward(self, x):
        x = self.layer_norm(x)
        x_prime = torch.concat([self_att(x) for self_att in self.attention_heads], axis=-1) # concatenate along data dim
        return self.output_projection(x_prime)+x