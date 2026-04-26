import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 若后续需要使用

"""
论文地址：https://arxiv.org/abs/2409.13435
论文题目：PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution (ACCV 2024)
"""

class SoftPooling2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        # 定义平均池化操作（不计入填充），用于计算指数加权平均
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        # 计算输入张量的指数
        x_exp = torch.exp(x)
        # 对指数化后的张量进行平均池化
        x_exp_pool = self.avgpool(x_exp)
        # 对输入乘以其指数，再进行平均池化
        x_weighted = self.avgpool(x_exp * x)
        # 返回归一化后的结果
        return x_weighted / x_exp_pool

class LocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        super(LocalAttention, self).__init__()
        # 定义主要的局部注意力处理流程
        self.body = nn.Sequential(
            # 通过1x1卷积调整通道数
            nn.Conv2d(channels, f, kernel_size=1),
            # 利用SoftPooling2D捕捉重要性信息
            SoftPooling2D(7, stride=3),
            # 3x3卷积（步长2）进一步提取局部特征
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            # 使用3x3卷积将通道恢复至原始数量
            nn.Conv2d(f, channels, kernel_size=3, padding=1),
            # Sigmoid激活生成空间权重
            nn.Sigmoid(),
        )
        # 定义简单的门控机制，防止伪影
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对输入第一通道进行门控操作
        g = self.gate(x[:, :1].clone())
        # 计算局部注意力权重，并通过双线性插值调整至原输入尺寸
        w = F.interpolate(self.body(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # 返回原始输入与局部注意力和门控信号的逐元素乘积
        return x * w * g

if __name__ == "__main__":
    # 创建随机输入张量，形状为 (1, 32, 64, 64)
    input = torch.randn(1, 32, 64, 64)
    # 实例化LocalAttention模块
    LA = LocalAttention(32)
    # 执行前向传播
    output = LA(input)
    # 打印输入和输出的尺寸
    print('input_size:', input.size())
    print('output_size:', output.size())
