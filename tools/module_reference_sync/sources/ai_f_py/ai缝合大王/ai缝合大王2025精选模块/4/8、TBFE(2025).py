import torch
import torch.nn as nn

"""
    论文来源：https://www.sciencedirect.com/science/article/pii/S1051200425000922
    论文标题：A synergistic CNN-transformer network with pooling attention fusion for hyperspectral image classification（2025）
"""

class TBFE(nn.Module):
    """
    双分支特征提取（Twin-branch Feature Extraction, TBFE）模块。
    该模块结合点卷积、深度可分离卷积和三维卷积，以提取丰富的空间-时序特征。
    """
    def __init__(self, input_channels, reduction_N=32):
        super(TBFE, self).__init__()
        
        # 1x1点卷积（通道降维）
        self.point_wise = nn.Conv2d(input_channels, reduction_N, kernel_size=1, bias=False)
        
        # 深度可分离卷积（空间特征提取）
        self.depth_wise = nn.Sequential(
            nn.Conv2d(reduction_N, reduction_N, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduction_N),
            nn.ReLU(),
        )
        
        # 3D卷积（时序特征建模）
        self.conv3D = nn.Conv3d(
            in_channels=1, out_channels=1, kernel_size=(1, 1, 3),
            padding=(0, 0, 1), stride=(1, 1, 1), bias=False
        )
        
        # 特征融合
        self.bn = nn.BatchNorm2d(2 * reduction_N)
        self.relu = nn.ReLU()
        
        # 1x1卷积（恢复原始通道数）
        self.pro = nn.Conv2d(2 * reduction_N, input_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        前向传播过程。
        """
        # 通道压缩
        x_1 = self.point_wise(x)
        
        # 空间特征提取（残差连接）
        x_2 = self.depth_wise(x_1)
        x_2 = x_1 + x_2
        
        # 时序特征建模
        x_3 = x_1.unsqueeze(1)  # 增加时序维度
        x_3 = self.conv3D(x_3).squeeze(1)
        
        # 特征融合
        x = torch.cat((x_2, x_3), dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pro(x)
        return x

if __name__ == "__main__":
    model = TBFE(input_channels=16)
    input_tensor = torch.randn(1, 16, 128, 128)
    output_tensor = model(input_tensor)
    
    print('输入张量尺寸:', input_tensor.size())
    print('输出张量尺寸:', output_tensor.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')