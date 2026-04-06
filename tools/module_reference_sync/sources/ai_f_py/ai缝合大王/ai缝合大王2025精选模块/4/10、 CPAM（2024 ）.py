import torch
import torch.nn as nn
import math
"""
    论文来源：https://arxiv.org/abs/2312.06458
    论文标题：ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation（2024 三区）
"""

# 通道注意力模块
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 局部注意力模块
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()
        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        return x * s_h.expand_as(x) * s_w.expand_as(x)

# CPAM 模块，结合通道注意力和局部注意力
class CPAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, input1, input2):
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x

if __name__ == '__main__':
    model = CPAM(ch=64)
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)
    output = model(input1, input2)
    
    print("输入张量 1 形状:", input1.shape)
    print("输入张量 2 形状:", input2.shape)
    print("输出张量 形状:", output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')