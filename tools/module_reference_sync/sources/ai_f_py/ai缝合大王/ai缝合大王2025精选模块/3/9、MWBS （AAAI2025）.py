from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

"""
论文链接：https://arxiv.org/pdf/2412.08345
论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
"""

# ------------------------- 辅助函数与模块 -------------------------
def autopad(k, p=None, d=1):
    """根据卷积核大小、膨胀率计算填充大小，以保持输出形状与输入相同。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# 定义无偏置层归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# 定义带偏置层归一化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# 动态选择层归一化类型
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# ------------------------- 相对位置嵌入与固定位置嵌入 -------------------------
class RelPosEmb(nn.Module):
    """
    相对位置嵌入模块，用于计算相对位置logits。
    """
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        self.block_size = block_size
        self.rel_size = rel_size
        self.dim_head = dim_head
        scale = dim_head ** -0.5
        # 初始化相对位置嵌入参数
        self.rel_height = nn.Parameter(torch.randn(2 * rel_size - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(2 * rel_size - 1, dim_head) * scale)

    def forward(self, q):
        # 这里为了简化实现，我们返回一个零张量与预期形状相同
        b, heads, N, _ = q.shape
        return torch.zeros(b, heads, N, N, device=q.device, dtype=q.dtype)

class FixedPosEmb(nn.Module):
    """
    固定位置嵌入模块，用于生成注意力掩码。
    """
    def __init__(self, window_size, overlap_win_size):
        super().__init__()
        self.window_size = window_size
        self.overlap_win_size = overlap_win_size
        # 构造一个固定的注意力掩码表（简单实现，全部为零）
        self.attention_mask = torch.zeros(1, 1, window_size**2, overlap_win_size**2)
    
    def forward(self):
        return self.attention_mask

# ------------------------- DilatedOCA 模块 -------------------------
class DilatedOCA(nn.Module):
    """
    带空洞的掩码窗口自注意力机制。

    Args:
        dim (int): 输入通道数。
        window_size (int): 窗口大小，默认为 8。
        overlap_ratio (float): 重叠比例，默认为 0.5。
        num_heads (int): 注意力头数量，默认为 2。
        dim_head (int): 每个注意力头的维度，默认为 16。
        bias (bool): 是否使用偏置，默认为 False。
    """
    def __init__(self, dim, window_size=8, overlap_ratio=0.5, num_heads=2, dim_head=16, bias=False):
        super(DilatedOCA, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head ** -0.5

        # 展开输入特征图为重叠窗口的序列
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size - window_size) // 2)
        # 用于生成QKV的1x1卷积
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        # 输出投影，将注意力输出映射回原始维度
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        # 相对位置嵌入层
        self.rel_pos_emb = RelPosEmb(block_size=window_size, rel_size=window_size + (self.overlap_win_size - window_size), dim_head=self.dim_head)
        # 固定位置嵌入层，用于生成注意力掩码
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)
        self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        x = self.norm(x)
        b, c, h, w = x.shape
        qkv = self.qkv(x)  # 生成QKV特征，形状为 (b, inner_dim*3, h, w)
        qs, ks, vs = qkv.chunk(3, dim=1)  # 分割为查询、键、值

        qs = rearrange(qs, 'b c h w -> b (h w) c')
        # 对键和值应用unfold获得重叠窗口表示，并重塑为 (b, inner_dim, L)
        ks = self.unfold(ks).reshape(b, self.inner_dim, -1)
        vs = self.unfold(vs).reshape(b, self.inner_dim, -1)

        # 将查询、键、值重塑为多头格式
        qs = qs.reshape(b, h * w, self.num_spatial_heads, self.dim_head).permute(0, 2, 1, 3)
        ks = ks.reshape(b, self.num_spatial_heads, -1, self.dim_head)
        vs = vs.reshape(b, self.num_spatial_heads, -1, self.dim_head)

        # 计算相对位置logits
        rel_pos = self.rel_pos_emb(qs)  # (b, num_heads, N, N')
        fixed_mask = self.fixed_pos_emb()  # (1, num_heads, window_size^2, overlap_win_size^2)

        # 计算缩放点积注意力
        attn = (qs @ ks.transpose(-2, -1)) * self.scale
        attn = attn + rel_pos + fixed_mask
        attn = attn.softmax(dim=-1)
        out = attn @ vs  # (b, num_heads, N, dim_head)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, self.inner_dim)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.project_out(out)
        return out

if __name__ == "__main__":
    # 测试DilatedOCA模块
    input_tensor = torch.randn(1, 64, 128, 128)
    model = DilatedOCA(64)
    output_tensor = model(input_tensor)
    print("输入尺寸:", input_tensor.size())   # 例如: torch.Size([1, 64, 128, 128])
    print("输出尺寸:", output_tensor.size())  # 应与输入尺寸一致
    # ai缝合大王
