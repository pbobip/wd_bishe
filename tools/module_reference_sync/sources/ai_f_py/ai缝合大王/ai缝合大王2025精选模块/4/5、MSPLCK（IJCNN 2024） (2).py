import torch
import torch.nn as nn
"""
    论文来源：https://arxiv.org/pdf/2305.17654
    论文标题：MixDehazeNet : Mix Structure Block For Image Dehazing Network（IJCNN 2024）
"""

class Multi_Scale_Parallel_Large_Convolution_Kernel_Model(nn.Module):
    """
    多尺度并行大卷积核模型。
    该模型采用多尺度卷积结构，包括 1x1、5x5 和不同扩张率的深度卷积，
    以增强特征提取能力。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)  # 批归一化

        # 基础卷积层
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1 卷积
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5 卷积

        # 深度扩张卷积（不同感受野）
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_5 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # 多层感知机（MLP）
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1 卷积升维
            nn.GELU(),  # GELU 激活
            nn.Conv2d(dim * 4, dim, 1)  # 1x1 卷积降维
        )

    def forward(self, x):
        """
        前向传播过程。
        """
        identity = x  # 残差连接
        x = self.norm1(x)  # 批归一化
        x = self.conv1(x)  # 1x1 卷积
        x = self.conv2(x)  # 5x5 卷积
        
        # 多尺度特征融合
        x = torch.cat([self.conv3_7(x), self.conv3_5(x), self.conv3_3(x)], dim=1)
        x = self.mlp(x)  # 通过 MLP 进行特征变换
        x = identity + x  # 残差连接
        return x

if __name__ == '__main__':
    # 创建模型实例
    model = Multi_Scale_Parallel_Large_Convolution_Kernel_Model(dim=32)
    input_tensor = torch.randn(8, 32, 64, 64)  # 生成随机输入
    output_tensor = model(input_tensor)  # 执行前向传播
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')
    
    print('输入张量尺寸:', input_tensor.size())
    print('输出张量尺寸:', output_tensor.size())