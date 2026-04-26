import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class ENLA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v):
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        return self.dropout(torch.matmul(attn, v))

class ENLTB(nn.Module):
    def __init__(self, dim, input_resolution, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ENLA(dim)
        self.drop_path = DropPath(0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.transpose(1, 2).view(B, C, H, W)


if __name__ == '__main__':
    input = torch.randn(4, 256, 56, 56)
    print("Input shape:", input.size())
    block = ENLTB(dim=256, input_resolution=(56, 56))
    output = block(input)
    print("Output shape:", output.size())
