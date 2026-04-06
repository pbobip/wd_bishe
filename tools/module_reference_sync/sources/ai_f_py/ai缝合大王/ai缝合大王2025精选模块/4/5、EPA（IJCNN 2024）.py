import torch
import torch.nn as nn

"""
    论文来源：https://arxiv.org/pdf/2305.17654
    论文标题：MixDehazeNet : Mix Structure Block For Image Dehazing Network（IJCNN 2024）
"""

class Enhanced_Parallel_Attention(nn.Module):
    """
    增强型并行注意力（EPA）模块。
    该模块结合了通道注意力（CA）、像素注意力（PA）和自注意力机制，以增强特征表达能力。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm2 = nn.BatchNorm2d(dim)  # 批归一化
        
        # 简单像素注意力
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')  # 深度可分离卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活
        )

        # 通道注意力（Channel Attention, CA）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dim, dim, 1, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活
            nn.Conv2d(dim, dim, 1, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活
        )

        # 像素注意力（Pixel Attention, PA）
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),  # 1x1卷积降维
            nn.GELU(),  # GELU激活
            nn.Conv2d(dim // 8, 1, 1, bias=True),  # 1x1卷积输出单通道
            nn.Sigmoid()  # Sigmoid激活
        )

        # MLP 层（用于融合不同注意力机制的输出）
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积升维
            nn.GELU(),  # GELU激活
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积降维
        )

    def forward(self, x):
        """
        前向传播过程。
        """
        identity = x  # 残差连接
        x = self.norm2(x)  # 批归一化
        # 拼接不同注意力机制的输出
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)  # 通过 MLP 进行通道融合
        x = identity + x  # 残差连接
        return x

if __name__ == '__main__':
    # 创建模型实例
    model = Enhanced_Parallel_Attention(dim=32)
    # 生成随机输入张量
    input_tensor = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output_tensor = model(input_tensor)
    
    print('输入张量尺寸:', input_tensor.size())  # 打印输入尺寸
    print('输出张量尺寸:', output_tensor.size())  # 打印输出尺寸
    
    # 计算模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')