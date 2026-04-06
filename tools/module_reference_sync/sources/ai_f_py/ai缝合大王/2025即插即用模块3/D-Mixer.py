import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qk_scale=None, attn_drop=0, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)

        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim, kernel_size=sr_ratio+3, stride=sr_ratio, padding=(sr_ratio+3)//2, groups=dim, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='GELU')),
                ConvModule(dim, dim, kernel_size=1, groups=dim, norm_cfg=dict(type='BN2d'), act_cfg=None),
            )
        else:
            self.sr = nn.Identity()

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, num_groups=1, bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.K = kernel_size
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size))
        self.pool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim, dim // reduction_ratio, kernel_size=1, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='GELU')),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )
        self.bias = nn.Parameter(torch.empty(num_groups, dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale_b = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True)).reshape(B, self.num_groups, C)
            scale_b = torch.softmax(scale_b, dim=1)
            bias = torch.sum(scale_b * self.bias.unsqueeze(0), dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W), weight=weight, padding=self.K // 2, groups=B * C, bias=bias)
        return x.reshape(B, C, H, W)

class HybridTokenMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1, reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0
        self.local_unit = DynamicConv2d(dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)
        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        return self.proj(x) + x
if __name__ == "__main__":
    # 示例输入：Batch size=1，通道数=64，图像大小=256x256
    x = torch.randn(1, 64, 256, 256)

    # 初始化 D-Mixer 模块（HybridTokenMixer）
    dmixer = HybridTokenMixer(dim=64, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1)

    # 打印模块结构
    print(dmixer)

    # 模块前向传播
    out = dmixer(x)

    # 输出结果形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
