import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Bilateral_Event_Mining_and_Complementary_for_Event_Stream_Super-Resolution_CVPR_2024_paper.pdf
论文标题：Bilateral Event Mining and Complementary for Event Stream Super-Resolution (CVPR 2024)
"""

def initialize_weights(net_l, scale=0.1):
    # 若传入的 net_l 不是列表，则将其转换为列表以便统一处理
    if not isinstance(net_l, list):
        net_l = [net_l]
    # 遍历网络中的每个子模块进行权重初始化
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 正态分布初始化卷积层权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # 缩放初始化权重
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置置零
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 对 BatchNorm 层进行常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        # 计算均值和方差（沿通道维度）以进行标准化
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        # 将标准化结果通过缩放和平移映射
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
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        # 注册权重和平移参数
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # 调用自定义的LayerNormFunction
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        # 定义两个3x3卷积层保持通道数不变
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x  # 保存输入以实现残差连接
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class BIE(nn.Module):
    def __init__(self, nf=64):
        super(BIE, self).__init__()
        # 两个并行残差块，作用于不同分支
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1  # 共享同一结构
        # 1x1卷积用于对拼接后的特征降维
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.convf2 = self.convf1
        self.scale = nf ** -0.5  # 缩放因子，用于注意力计算
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)
        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2, x_s):
        b, c, h, w = x_1.shape
        # 分别处理两个输入分支
        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        # 计算共享类别中心，利用拼接和归一化进行特征整合
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1)
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1)
        # 对输入进行特征映射并调整维度便于矩阵乘法
        v_1 = self.v1(x_1).view(b, c, -1).permute(0, 2, 1)
        v_2 = self.v2(x_2).view(b, c, -1).permute(0, 2, 1)
        # 计算注意力得分
        att1 = torch.bmm(shared_class_center1, v_1) * self.scale
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale
        # 通过 softmax 归一化注意力得分后获得加权特征
        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)
        # 融合解聚类操作和原始特征
        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w),
                                              shared_class_center2.view(b, c, h, w)], dim=1)) + x_s
        # 返回经过注意力增强的分支输出与融合后的辅助特征
        return out_1 + x_2_, out_2 + x_1_, x_s_

if __name__ == '__main__':
    x1 = torch.randn(1, 32, 64, 64)
    x2 = torch.randn(1, 32, 64, 64)
    x3 = torch.randn(1, 32, 64, 64)
    Model = BIE(nf=32)
    out = Model(x1, x2, x3)
    print("Input size:", x1.size(), x2.shape, x3.shape)
    print("Output size:", out[0].size())
    print("Output size:", out[1].size())
    print("Output size:", out[2].size())
    # ai缝合大王
