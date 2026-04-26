import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
论文链接：https://arxiv.org/pdf/2412.08345  
论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
"""

# CBR模块：卷积 -> 批归一化 -> 可选ReLU激活
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 控制是否激活
        # 构建卷积和归一化组合层
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 使用ReLU激活函数
        # ai缝合大王

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

# 通道注意力模块：结合全局平均和最大池化，生成通道权重
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)    # 全局最大池化

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)  # 通道降维
        self.relu1 = nn.ReLU()  # 激活
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)  # 通道升维

        self.sigmoid = nn.Sigmoid()  # 映射到 [0, 1]

    def forward(self, x):
        x_orig = x  # 保留原始输入
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x_orig * self.sigmoid(out)

# 空间注意力模块：利用通道统计生成空间权重图
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        # 核尺寸必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # 将卷积输出映射至 [0,1]
        # ai缝合大王

    def forward(self, x):
        x_orig = x  # 保存原始特征
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对通道取平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对通道取最大值
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_conv = self.conv1(x_cat)
        return x_orig * self.sigmoid(x_conv)

# 膨胀卷积模块：采用不同膨胀率的卷积以捕捉多尺度上下文信息，并通过注意力机制进行特征重构
class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # 每个分支先用CBR模块做卷积，再结合通道注意力
        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=3, padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=3, padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=3, padding=18, dilation=18), channel_attention(out_c))
        # 合并分支后使用3x3卷积整合多尺度信息
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        # 直接1x1卷积对输入进行映射
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()  # 空间注意力模块

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        # 将四个分支输出在通道维度上拼接
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x_out = self.relu(xc + xs)  # 残差连接和激活
        x_out = self.sa(x_out)  # 空间注意力加权
        return x_out

if __name__ == '__main__':
    input_tensor = torch.rand(1, 64, 32, 32)  # 随机生成输入张量
    drm = dilated_conv(64, 64)  # 实例化膨胀卷积模块
    output_tensor = drm(input_tensor)
    print("Input size:", input_tensor.size())
    print("Output size:", output_tensor.size())
    # ai缝合大王
