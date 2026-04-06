# 导入必要的库
import torch.nn.functional as F  # PyTorch神经网络函数库
from einops import rearrange  # 张量维度操作库
import numbers  # 数字类型判断库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块

"""
论文地址：https://arxiv.org/abs/2404.07846
论文题目：Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising（AAAI 2025）
"""

# 张量维度转换函数 ============================================================
def to_3d(x):
    """将四维张量[B, C, H, W]转换为三维[B, H*W, C]格式（用于序列处理）"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """将三维张量[B, H*W, C]恢复为四维[B, C, H, W]格式（恢复空间结构）"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# 层归一化模块 =============================================================
class BiasFree_LayerNorm(nn.Module):
    """无偏置的层归一化实现（仅含可学习缩放参数）"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化实现（包含缩放和偏移参数）"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """动态选择归一化类型的层归一化模块"""
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# 核心注意力模块 ============================================================
# Grouped_Channel_SelfAttention
class Grouped_Channel_SelfAttention(nn.Module):
    """带空洞的多头通道注意力机制"""
    def __init__(self, dim, num_heads=2, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    dilation=2, padding=2, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

    def forward(self, x):
        x = self.norm(x)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))  # 生成 QKV 特征
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        return self.project_out(out)

# 模块测试 =================================================================
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 128, 128)
    DilatedMDTA_channel_attn = Grouped_Channel_SelfAttention(64)
    output_tensor = DilatedMDTA_channel_attn(input_tensor)
    print('输入尺寸:', input_tensor.size())
    print('输出尺寸:', output_tensor.size())
    # ai缝合大王
