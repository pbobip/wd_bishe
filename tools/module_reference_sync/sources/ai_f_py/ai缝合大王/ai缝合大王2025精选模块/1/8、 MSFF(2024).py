import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文链接：https://ieeexplore.ieee.org/abstract/document/10445289/
论文标题：Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024)
"""

class Multi_Scale_Feed_Forward_Network(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Multi_Scale_Feed_Forward_Network, self).__init__()

        # 根据扩展因子计算隐藏层的通道数
        hidden_features = int(dim * ffn_expansion_factor)

        # 通过3D卷积将输入特征映射到扩展后的隐藏空间
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)

        # 第一层深度可分离3D卷积，核尺寸为 3x3x3，确保空间尺寸保持不变
        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3),
                                 stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)

        # 第二层使用2D深度可分离卷积（空洞卷积），空洞率设置为2，捕捉中尺度特征
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3),
                                 stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)

        # 第三层2D深度可分离卷积，空洞率为3，用于捕捉更大范围的上下文信息
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3),
                                 stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)

        # 利用3D卷积将处理后的特征映射回原始通道数
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)

    def forward(self, x):
        # 在第2维添加一个维度以满足3D卷积的输入要求
        x = x.unsqueeze(2)  # 形状变为 [B, C, 1, H, W]
        # 通过 project_in 将输入扩展到隐藏空间
        x = self.project_in(x)  # 输出形状为 [B, 3*hidden_features, 1, H, W]
        # 将扩展后的通道分成三组，各自用于后续不同的卷积操作
        x1, x2, x3 = x.chunk(3, dim=1)  # 每组形状为 [B, hidden_features, 1, H, W]
        # 对第一组应用3D深度卷积，然后移除添加的维度
        x1 = self.dwconv1(x1).squeeze(2)
        # 对第二和第三组，先移除额外维度，再利用2D空洞卷积分别处理
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        # 使用 GELU 激活函数对 x1 进行非线性映射，再与 x2 和 x3 逐元素相乘
        x = F.gelu(x1) * x2 * x3
        # ai缝合大王
        # 为输出重新添加一个深度维度以适应 project_out 的输入要求
        x = x.unsqueeze(2)
        # 通过 project_out 将特征映射回原始维度
        x = self.project_out(x)
        # 去除额外的深度维度，恢复到二维空间结构
        x = x.squeeze(2)
        # ai缝合大王
        return x

# 主程序入口
if __name__ == '__main__':
    # 生成形状为 (1, 32, 8, 8) 的随机输入张量
    input = torch.randn(1, 32, 8, 8)
    # 实例化多尺度前馈网络，设置输入维度为32，扩张因子为2，并开启偏置项
    model = Multi_Scale_Feed_Forward_Network(dim=32, ffn_expansion_factor=2, bias=True)
    # 执行前向传播，获取输出张量
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
    # ai缝合大王
