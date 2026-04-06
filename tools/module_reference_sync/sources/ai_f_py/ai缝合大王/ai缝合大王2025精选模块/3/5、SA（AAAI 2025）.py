import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
论文链接：https://arxiv.org/pdf/2412.08345
论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
"""

# CBR模块：卷积 -> 批归一化 -> （可选）ReLU激活
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 控制是否使用激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

# 通道注意力模块：通过全局平均池化和最大池化生成通道权重
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出尺寸为1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)    # 自适应最大池化，输出尺寸为1x1

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)  # 降维
        self.relu1 = nn.ReLU()  # 激活
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)  # 升维

        self.sigmoid = nn.Sigmoid()  # Sigmoid将输出映射到[0,1]

    def forward(self, x):
        x0 = x  # 保存原始输入
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化分支
        out = avg_out + max_out  # 融合分支
        return x0 * self.sigmoid(out)  # 对输入加权后返回

# 空间注意力模块：通过对通道的平均与最大池化生成空间注意力图
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # 保持输入原始特征 [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大池化
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 拼接池化结果
        x_conv = self.conv1(x_cat)  # 生成空间注意力图
        return x0 * self.sigmoid(x_conv)  # 对输入进行加权

# 膨胀卷积模块：利用不同扩张率的卷积捕捉多尺度信息，并通过注意力机制进行特征融合
class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.c1 = nn.Sequential(
            CBR(in_c, out_c, kernel_size=1, padding=0),
            channel_attention(out_c)
        )
        self.c2 = nn.Sequential(
            CBR(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            channel_attention(out_c)
        )
        self.c3 = nn.Sequential(
            CBR(in_c, out_c, kernel_size=3, padding=12, dilation=12),
            channel_attention(out_c)
        )
        self.c4 = nn.Sequential(
            CBR(in_c, out_c, kernel_size=3, padding=18, dilation=18),
            channel_attention(out_c)
        )
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)  # 第一个膨胀卷积分支
        x2 = self.c2(x)  # 第二个膨胀卷积分支
        x3 = self.c3(x)  # 第三个膨胀卷积分支
        x4 = self.c4(x)  # 第四个膨胀卷积分支
        xc = torch.cat([x1, x2, x3, x4], axis=1)  # 拼接所有分支
        xc = self.c5(xc)  # 混合卷积整合多尺度特征
        xs = self.c6(x)   # 直接通过1x1卷积处理输入
        x_out = self.relu(xc + xs)  # 残差连接并激活
        x_out = self.sa(x_out)  # 应用空间注意力
        return x_out

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    drm = dilated_conv(64, 64)
    output = drm(input)
    print("DRM_input.shape:", input.shape)
    print("DRM_output.shape:", output.shape)
    # ai缝合大王
