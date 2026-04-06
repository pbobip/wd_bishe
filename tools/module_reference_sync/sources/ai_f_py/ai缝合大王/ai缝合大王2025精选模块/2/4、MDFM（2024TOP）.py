import torch
import torch.nn as nn
import torch.nn.functional as F

'''
论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
论文题目：Dual Residual Attention Network for Image Denoising （2024）
'''

# 多尺度特征融合模块 (MSFF)
class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        # 定义不同尺寸的卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 混合卷积层用于融合各尺度卷积输出
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 分别计算各尺度卷积输出
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        # 拼接并混合处理
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)
        return out

# 多尺度差异融合模块 (MDFM)
class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # 使用 MSFF 模块实现多尺度特征融合
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64)
        # 差异增强卷积模块
        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        # 降维卷积模块，将通道数从 in_d 降到 out_d
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        # 差异提取卷积模块
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )
        # 混合卷积模块（采用深度卷积）用于整合两路特征
        self.convmix = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, groups=self.in_d, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # 计算输入间的绝对差异
        x_sub = torch.abs(x1 - x2)
        # 提取差异特征并生成注意力权重
        x_att = torch.sigmoid(self.conv_sub(x_sub))
        # 利用注意力权重调制各输入，并加入多尺度特征融合结果
        x1 = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2 = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))
        # 将两个输入沿新维度堆叠后重塑
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (x1.size(0), -1, x1.size(2), x1.size(3)))
        # 采用混合卷积融合特征
        x_f = self.convmix(x_f)
        # 利用注意力权重进一步加权特征
        x_f = x_f * x_att
        out = self.conv_dr(x_f)
        return out

if __name__ == '__main__':
    # 创建两个随机输入张量，尺寸分别为 (32, 512, 8, 8)
    x1 = torch.randn((32, 512, 8, 8))
    x2 = torch.randn((32, 512, 8, 8))
    # 实例化 MDFM 模块，将输入通道数设为 512，输出通道数设为 64
    model = MDFM(512, 64)
    # 前向传播得到输出
    out = model(x1, x2)
    print(out.shape)
