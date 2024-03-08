import torch
import math

import torch.nn.functional as F
from torch import Tensor
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


# Patch Embedding: 2D Convolution ksize = (channels, 31), stride = (1, 31)
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, emb_size, (19, 64), stride=(1, 64)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.emb_size = emb_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        # shape of positions depends on sample size (10 sec * 512 Hz)

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, s = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        positions = nn.Parameter(torch.randn((s // 64) + 1, self.emb_size)).cuda()
        x += positions
        return x


# MultiHeadAttention (see paper)
class MultiHeadAttention(nn.Module):
    def __init__(self, i, emb_size, num_heads, dropout):
        super().__init__()
        self.i = i
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        #save attention weights for interpretability
        torch.save(att, 'att' + str(self.i) +'.pt')
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 i,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=2,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(i, emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, i, depth, emb_size, num_heads):
        super().__init__(*[TransformerEncoderBlock(K, emb_size, num_heads) for K in range(depth)])


# ClassificationHead - 1 linear layer
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            #Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x[:, 0])
       # out = self.clshead(x)
        return x, out


# ViT - working version: 32 embeddings, 8 heads, depth 2
class ViT(nn.Sequential):
    def __init__(self, i=0, emb_size=32, num_heads=1, depth= 2, n_classes=3, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(i, depth, emb_size, num_heads),
            ClassificationHead(emb_size, n_classes)
        )
