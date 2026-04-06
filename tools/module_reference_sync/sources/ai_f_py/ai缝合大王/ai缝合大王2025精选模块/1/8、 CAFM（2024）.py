import torch
import torch.nn as nn
from einops import rearrange

"""
论文链接：https://ieeexplore.ieee.org/abstract/document/10445289/
论文题目：Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024)
"""

class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()  # 初始化父类模块
        self.num_heads = num_heads  # 指定多头注意力中的头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 初始化温度参数，用于缩放点积

        # 定义用于生成查询、键和值的 1x1x1 卷积（3D卷积）
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # 对 qkv 结果应用 3D 深度可分离卷积以捕捉局部空间关系
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1,
                                     groups=dim * 3, bias=bias)
        # 输出投影层，将多头注意力结果映射回原始通道数
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

        # 全连接层用于调整通道数，将通道数从 3*num_heads 调整到 9
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        # 深度卷积层用于提取局部细节特征，其分组数使得每个组只处理部分通道
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
                                  bias=True, groups=dim // self.num_heads, padding=1)
        # ai缝合大王

    def forward(self, x):
        b, c, h, w = x.shape  # x 的形状为 (B, C, H, W)
        x = x.unsqueeze(2)  # 在第三个维度插入单一深度维度，形状变为 (B, C, 1, H, W)
        qkv = self.qkv_dwconv(self.qkv(x))  # 首先计算 qkv，然后通过深度卷积进一步提取特征
        qkv = qkv.squeeze(2)  # 移除深度维度，恢复至 (B, C, H, W)

        # ========= 局部特征分支 =========
        # 将张量转换为 (B, H, W, C) 以便后续全连接层处理
        f_conv = qkv.permute(0, 2, 3, 1)
        # 重塑张量以组织多头特征，通道维度分为 3*num_heads 和剩余特征维度
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))  # 应用全连接层调整特征通道
        f_all = f_all.squeeze(2)
        # 调整维度以适应局部卷积层的输入
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # 利用深度卷积提取局部细节信息
        out_conv = out_conv.squeeze(2)

        # ========= 全局特征分支 =========
        # 将 qkv 分成查询、键和值三个部分
        q, k, v = qkv.chunk(3, dim=1)
        # 重排张量形状，转换为 (B, head, c, (H*W)) 以便于多头注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 对查询和键进行 L2 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算缩放后的点积注意力，并归一化
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 用注意力权重加权值向量
        out = (attn @ v)
        # 将多头输出重排回原始空间尺寸 (B, C, H, W)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)  # 进行输出投影
        out = out.squeeze(2)

        # 将局部与全局分支结果相加，融合多尺度特征
        output = out + out_conv
        return output

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)  # 随机生成输入张量
    cafm = CAFM(dim=64)
    output = cafm(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
    # ai缝合大王
