import torch
import torch.nn as nn
from einops import rearrange

"""
    论文来源：https://arxiv.org/abs/2501.15061
    论文标题：PolaFormer: Polarity-aware Linear Attention for Vision Transformers (ICLR 2025)
"""

class PolaLinearAttention(nn.Module):
    """
    极性感知线性注意力（PolaLinearAttention）。
    该模块引入可学习的幂次增强函数和深度可分离卷积，以提升注意力机制的表达能力。
    """
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # 生成查询和门控向量
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 生成键和值
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 空间缩减卷积
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
        # 深度可分离卷积
        self.dwc = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=kernel_size,
                             groups=self.head_dim, padding=kernel_size // 2)
        
        # 可学习幂次增强参数
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        self.kernel_function = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        
        k, v = kv[0], kv[1]
        n = k.shape[1]
        k = k + self.positional_encoding
        
        scale = nn.Softplus()(self.scale)
        q, k = q / scale, k / scale
        
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        
        power = 1 + self.alpha * torch.sigmoid(self.power)
        q_pos, q_neg = self.kernel_function(q) ** power, self.kernel_function(-q) ** power
        q_sim, q_opp = torch.cat([q_pos, q_neg], dim=-1), torch.cat([q_neg, q_pos], dim=-1)
        k_pos, k_neg = self.kernel_function(k) ** power, self.kernel_function(-k) ** power
        k = torch.cat([k_pos, k_neg], dim=-1)
        v1, v2 = torch.chunk(v, 2, dim=-1)
        
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z
        
        x = torch.cat([x_sim, x_opp], dim=-1).transpose(1, 2).reshape(B, N, C)
        
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n),
                                          size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        
        x = x + v
        x = x * g
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 维度转换函数

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

if __name__ == '__main__':
    B, C, H, W = 2, 128, 8, 8
    N = H * W
    tensor_4d = torch.randn(B, C, H, W)
    input_tensor = to_3d(tensor_4d)
    block = PolaLinearAttention(dim=C, num_patches=N, sr_ratio=1)
    output = block(input_tensor, H, W)
    output = to_4d(output, H, W)
    
    print("输入尺寸:", tensor_4d.size())
    print("输出尺寸:", output.size())