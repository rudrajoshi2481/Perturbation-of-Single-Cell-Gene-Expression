import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, embeding_dim=512, dropout=0.1, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embeding_dim // n_heads

        self.q = nn.Linear(embeding_dim, embeding_dim)
        self.k = nn.Linear(embeding_dim, embeding_dim)
        self.v = nn.Linear(embeding_dim, embeding_dim)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embeding_dim, embeding_dim),
            nn.ReLU(),
            nn.Linear(embeding_dim, embeding_dim)
        )

        self.prenorm = nn.LayerNorm(embeding_dim)

    def _reshape_heads(self, x):
        b, s, e = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        b, s, e = x.shape

        x = self.prenorm(x)
        residual = x

        q = self._reshape_heads(self.q(x))
        k = self._reshape_heads(self.k(x))
        v = self._reshape_heads(self.v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context_vec = torch.matmul(scores, v)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, s, e)

        x = self.ffn(context_vec)
        return x + residual
