"""
cross_attention.py — reconstructed for DSA-Mamba
(original .py was missing from repo; reconstructed from bytecode analysis)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DWConv(nn.Module):
    """Depthwise 2-D convolution."""
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                groups=dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.size()
        tx = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        conv_x = self.dwconv(tx)          # (B, C, H, W)
        return conv_x.flatten(2).transpose(1, 2).view(B, H, W, C)


class skip_ffn(nn.Module):
    """Skip-connection FFN with depthwise conv."""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.fc1   = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c1)
        self.norm3 = nn.LayerNorm(c1)

    def forward(self, x: Tensor) -> Tensor:
        ax  = self.act(self.norm1(self.dwconv(self.fc1(x))))
        out = self.fc2(ax)
        return self.norm3(out + x)


class Cross_Attention(nn.Module):
    """
    Multi-head cross-attention using Conv2d projections.
    Operates on spatial feature maps in (B, H, W, C) format.
    """
    def __init__(self, key_channels: int, value_channels: int, head_count: int = 1):
        super().__init__()
        self.key_channels   = key_channels
        self.head_count     = head_count
        self.value_channels = value_channels

        # Project back to key_channels after multi-head aggregation
        self.reprojection = nn.Conv2d(value_channels, key_channels, kernel_size=1)
        self.norm = nn.LayerNorm(key_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        x1: query source  (B, H, W, C)
        x2: key/value src (B, H, W, C)
        """
        B, H, W, C = x1.size()

        # Flatten spatial dims: (B, H*W, C) → (B, C, H*W)
        keys    = x2.view(B, -1, C).transpose(1, 2)   # (B, C, L)
        queries = x1.view(B, -1, C).transpose(1, 2)   # (B, C, L)
        values  = x2.view(B, -1, C).transpose(1, 2)   # (B, C, L)

        head_key_channels   = self.key_channels   // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key   = F.softmax(keys  [:, i*head_key_channels  :(i+1)*head_key_channels,   :], dim=2)
            query = F.softmax(queries[:, i*head_key_channels :(i+1)*head_key_channels,   :], dim=1)
            value = values[:, i*head_value_channels:(i+1)*head_value_channels, :]           # (B, dv, L)

            # context = key^T @ value  →  (B, dk, dv)
            context = torch.einsum('bdk,bvk->bdv', key, value)  # (B, dk, dv)
            attended_value = torch.einsum('bdv,bdl->bvl', context, query)  # (B, dv, L)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)                # (B, C_v, L)
        aggregated_values = aggregated_values.reshape(B, -1, H, W)          # (B, C_v, H, W)
        reprojected_value = self.reprojection(aggregated_values)             # (B, C_k, H, W)
        reprojected_value = reprojected_value.permute(0, 2, 3, 1)           # (B, H, W, C_k)
        return self.norm(reprojected_value)


class CrossAttention(nn.Module):
    """
    Full cross-attention block with residual and skip-FFN.
    Used in DSA-Mamba decoder layers.

    Args:
        in_dim    : input channel dimension
        key_dim   : key/query dimension for inner Cross_Attention
        value_dim : value dimension for inner Cross_Attention
        head_count: number of attention heads
        token_mlp : whether to use token MLP (skip_ffn)
    """
    def __init__(self, in_dim: int, key_dim: int, value_dim: int,
                 head_count: int = 1, token_mlp: bool = True):
        super().__init__()
        self.norm1    = nn.LayerNorm(in_dim)
        self.linear   = nn.Linear(in_dim, key_dim)
        self.attn     = Cross_Attention(key_dim, value_dim, head_count)
        # project attention output (key_dim) back to in_dim for residual connection
        self.out_proj = nn.Linear(key_dim, in_dim) if key_dim != in_dim else nn.Identity()
        self.norm2    = nn.LayerNorm(in_dim)
        self.mlp      = skip_ffn(in_dim, int(in_dim * 2)) if token_mlp else nn.Identity()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        x1: decoder input  (B, H, W, in_dim)
        x2: encoder skip   (B, H, W, in_dim)  [projected to in_dim before calling]
        Returns: (B, H, W, in_dim)
        """
        # Normalize both inputs and project down to key_dim for attention
        n1 = self.linear(self.norm1(x1))    # (B, H, W, key_dim)
        n2 = self.linear(self.norm1(x2))    # (B, H, W, key_dim)

        attn = self.attn(n1, n2)            # (B, H, W, key_dim)
        attn = self.out_proj(attn)          # (B, H, W, in_dim)

        tx = attn + x1                      # residual in in_dim space

        if isinstance(self.mlp, skip_ffn):
            return self.mlp(self.norm2(tx))
        return tx
