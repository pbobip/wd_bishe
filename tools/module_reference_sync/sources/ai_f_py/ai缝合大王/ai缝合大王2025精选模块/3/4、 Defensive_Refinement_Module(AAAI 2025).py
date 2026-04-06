import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Bilateral_Event_Mining_and_Complementary_for_Event_Stream_Super-Resolution_CVPR_2024_paper.pdf
论文题目：Bilateral Event Mining and Complementary for Event Stream Super-Resolution (CVPR 2024)
"""

def initialize_weights(net_l, scale=0.1):
    # 如果输入不是列表，则将其转换为列表
    if not isinstance(net_l, list):
        net_l = [net_l]
    # 遍历网络中的每个模块进行权重初始化
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态分布初始化卷积层权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # 缩放权重
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置设为零
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化BatchNorm层
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        # 计算均值和方差（沿通道维度）
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)
        # 应用可学习的缩放和平移
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # ai缝合大王
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)
        return gx, grad_weight, grad_bias, None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))  # 权重参数
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))  # 偏置参数
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # 第一个卷积层
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # 第二个卷积层
        initialize_weights([self.conv1, self.conv2], 0.1)  # 初始化卷积层权重

    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out  # 返回残差连接结果

class BIE(nn.Module):
    def __init__(self, in_channels):
        super(BIE, self).__init__()
        self.conv1 = ResidualBlock_noBN(in_channels)
        self.conv2 = ResidualBlock_noBN(in_channels)
        self.convf1 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, padding=0)
        self.convf2 = self.convf1

        self.scale = in_channels ** -0.5
        self.norm_s = LayerNorm2d(in_channels)
        self.clustering = nn.Conv2d(in_channels, in_channels, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)

        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2, x_s):
        b, c, h, w = x_1.shape
        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        # 计算共享类别中心
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1)
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1)
        # 提取特征映射，并调整形状便于矩阵乘法
        v_1 = self.v1(x_1).view(b, c, -1).permute(0, 2, 1)
        v_2 = self.v2(x_2).view(b, c, -1).permute(0, 2, 1)
        att1 = torch.bmm(shared_class_center1, v_1) * self.scale
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale
        # 计算注意力加权后的输出特征
        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)
        # 融合解聚类操作与辅助特征
        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w),
                                              shared_class_center2.view(b, c, h, w)], dim=1)) + x_s
        return out_1 + x_2_, out_2 + x_1_, x_s_

if __name__ == '__main__':
    x1 = torch.randn(1, 32, 64, 64)
    x2 = torch.randn(1, 32, 64, 64)
    x3 = torch.randn(1, 32, 64, 64)
    Model = BIE(in_channels=32)
    out = Model(x1, x2, x3)
    print("Input size:", x1.size(), x2.size(), x3.size())
    print("Output size:", out[0].size())
    print("Output size:", out[1].size())
    print("Output size:", out[2].size())
    # ai缝合大王
