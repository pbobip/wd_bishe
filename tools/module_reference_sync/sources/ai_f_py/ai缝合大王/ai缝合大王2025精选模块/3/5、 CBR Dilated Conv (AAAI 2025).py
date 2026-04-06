import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文链接：https://arxiv.org/pdf/2407.21640
论文题目：MSA2Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation (BMVA 2024)
"""

class GlobalExtraction(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        # 使用全局平均池化提取通道平均特征
        self.avgpool = self.globalavgchannelpool
        # 使用全局最大池化提取通道极值特征
        self.maxpool = self.globalmaxchannelpool
        # 利用1x1卷积和批归一化融合平均与最大池化的特征
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1)
        )

    def globalavgchannelpool(self, x):
        return x.mean(1, keepdim=True)

    def globalmaxchannelpool(self, x):
        return x.max(dim=1, keepdim=True)[0]

    def forward(self, x):
        x_clone = x.clone()
        x_avg = self.avgpool(x)    # 平均池化
        x_max = self.maxpool(x_clone)  # 最大池化
        cat = torch.cat((x_avg, x_max), dim=1)
        proj = self.proj(cat)
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super().__init__()
        self.reduction = 1 if reduction is None else 2
        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv2d(dim, dim // self.reduction, kernel_size=1),
            nn.BatchNorm2d(dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)
        x = self.proj(x)
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x, g):
        x_local = self.local(x)
        g_global = self.global_(g)
        fuse = self.bn(x_local + g_global)
        return fuse

class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
        )

    def forward(self, x, g):
        x_orig = x.clone()
        g_orig = g.clone()

        # 多尺度融合（局部与全局信息）
        multi = self.multi(x, g)
        # 自适应特征选择
        multi = self.selection(multi)
        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)
        x_att = A.expand_as(x_orig) * x_orig
        g_att = B.expand_as(g_orig) * g_orig
        x_att = x_att + x_orig
        g_att = g_att + g_orig

        # 特征交互与增强
        x_sig = torch.sigmoid(x_att)
        g_att_mod = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att_mod = g_sig * x_att
        interaction = x_att_mod * g_att_mod

        # 特征重校准
        projected = torch.sigmoid(self.bn(self.proj(interaction)))
        weighted = projected * x_orig
        y = self.conv_block(weighted)
        y = self.bn_2(y)
        return y

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    Model = MultiScaleGatedAttn(dim=64)
    out = Model(x1, x2)
    print(out.shape)
    print("Input size:", x1.size(), x2.size())
    print("Output size:", out.size())
    # ai缝合大王
