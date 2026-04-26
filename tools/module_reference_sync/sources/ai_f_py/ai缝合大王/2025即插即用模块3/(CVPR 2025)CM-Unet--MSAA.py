import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)

        # 通道注意力
        x_fused_c = x_fused * self.channel_attention(x_fused)

        # 多尺度卷积
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7

        # 空间注意力
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        # 输出融合
        x_out = self.up(x_fused_s + x_fused_c)
        return x_out


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        MSAA: Multi-Scale Attention Aggregation Module
        输入通道 = x1 + x2 + x4 三路拼接后的通道数
        输出通道 = out_channels
        """
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2, x4, last=False):
        """
        参数:
            x1: 当前主分支特征
            x2: 上采样来的特征（语义信息）
            x4: 下采样来的特征（边缘或细节）
        返回:
            融合后的特征
        """
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused
if __name__ == "__main__":
    # 模拟3路特征输入，每路通道为64，空间大小为56×56
    B, C, H, W = 4, 64, 56, 56
    x1 = torch.randn(B, C, H, W).cuda()
    x2 = torch.randn(B, C, H, W).cuda()
    x4 = torch.randn(B, C, H, W).cuda()

    # 构建 MSAA 模块：输入为 3×C，输出仍为 C
    msaa_block = MSAA(in_channels=3 * C, out_channels=C).cuda()
    output = msaa_block(x1, x2, x4)

    print(f"Input shape: x1={x1.shape}, x2={x2.shape}, x4={x4.shape}")
    print(f"Output shape: {output.shape}")
