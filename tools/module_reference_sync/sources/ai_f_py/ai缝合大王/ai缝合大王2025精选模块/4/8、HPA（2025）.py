import torch
import torch.nn as nn

"""
    论文来源：https://www.sciencedirect.com/science/article/pii/S1051200425000922
    论文标题：A synergistic CNN-transformer network with pooling attention fusion for hyperspectral image classification（2025）
"""

class HPA(nn.Module):
    """
    混合池化注意力（Hybrid Pooling Attention, HPA）模块。
    该模块结合全局池化（平均池化和最大池化）和通道归一化，以增强特征表达能力。
    """
    def __init__(self, channels, factor=32):
        super(HPA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0

        # 双池化分支
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))

        # 空间池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))

        # 特征变换层
        self.gn = nn.GroupNorm(num_groups=channels // self.groups, num_channels=channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)

        # 平均池化分支
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 最大池化分支
        y_h = self.max_h(group_x)
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))
        y_h, y_w = torch.split(yhw, [h, w], dim=2)
        y1 = self.gn(group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid())

        # 注意力权重融合
        x11 = x1.reshape(b * self.groups, -1, h * w)
        x12 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        y11 = y1.reshape(b * self.groups, -1, h * w)
        y12 = self.softmax(self.map(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

if __name__ == "__main__":
    model = HPA(64)
    input_tensor = torch.randn(2, 64, 128, 128)
    output_tensor = model(input_tensor)
    
    print('输入张量尺寸:', input_tensor.size())
    print('输出张量尺寸:', output_tensor.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')
