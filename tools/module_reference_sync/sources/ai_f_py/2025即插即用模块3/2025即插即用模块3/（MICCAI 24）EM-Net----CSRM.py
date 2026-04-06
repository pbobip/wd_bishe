import torch
import torch.nn as nn


class CSRM(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_norm = self.norm(x_flat).permute(0, 2, 1).view(B, C, H, W)
        x_mamba = self.mamba(x_norm)
        x_spatial = self.mlp(x_norm)
        return x + self.gamma * (x_mamba + x_spatial)

if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = CSRM(dim=32)
    out = model(x)
    print("✅ CSRM 测试输出形状:", out.shape)