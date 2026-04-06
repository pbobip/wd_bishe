import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

"""
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
论文题目：RMT: Retentive Networks Meet Vision Transformers  (CVPR 2024)
"""

class RetNetRelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        # 计算位置编码的角度，基于论文中常用的公式
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # 计算衰减因子，这里通过对初始值和头部范围进行对数运算得到
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = mask.abs().sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(slen[0], slen[1], -1)
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = ((sin, cos), (mask_h, mask_w))
        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = ((sin, cos), mask)
        return retention_rel_pos

def rotate_every_two(x):
    # 分别取出奇数和偶数索引处的元素，并执行旋转操作
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    out = x.flatten(-2)
    return out

def theta_shift(x, sin, cos):
    # 使用sin和cos对输入特征进行位置调制
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        # 初始化深度卷积，groups=dim确保每个通道独立处理
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        # 转换张量形状为 (B, C, H, W) 进行卷积
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class VisionRetentionChunk(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos  # 提取相对位置编码和衰减掩码

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)  # 获取局部增强特征

        k *= self.scaling
        # 调整查询和键的形状以适配多头注意力机制
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        # 应用位置调制
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        # 针对宽度方向计算注意力
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v = torch.matmul(qk_mat_w, v)
        # 针对高度方向计算注意力
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = torch.matmul(qk_mat_h, v)
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

if __name__ == '__main__':
    # 注意：输入张量形状为 B x H x W x C (非标准的NCHW格式)
    inputs = torch.randn(1, 32, 32, 64)  # 生成随机输入张量
    b, h, w, c = inputs.size()
    pos = RetNetRelPos2d(embed_dim=64, num_heads=4, initial_value=1, heads_range=3)
    # 计算相对位置编码，采用chunkwise_recurrent策略
    rel_pos = pos((h, w), chunkwise_recurrent=True)
    print(rel_pos)
    Model = VisionRetentionChunk(embed_dim=64, num_heads=4)
    out = Model(inputs, rel_pos)
    print(out.shape)
    # ai缝合大王
