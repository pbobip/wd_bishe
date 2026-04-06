import torch
import torch.nn as nn
import math
from math import log
from typing import List
#论文：LEGNet: Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection
#论文地址：https://arxiv.org/pdf/2503.14012
# 可选：替换为你自己的 build_norm_layer 方法
def build_norm_layer(norm_cfg, num_features):
    norm_type = norm_cfg.get('type', 'BN')
    requires_grad = norm_cfg.get('requires_grad', True)
    if norm_type == 'BN':
        layer = nn.BatchNorm2d(num_features)
    else:
        raise NotImplementedError
    for p in layer.parameters():
        p.requires_grad = requires_grad
    return norm_type, layer


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, 64, 1),
            build_norm_layer(norm_layer, 64)[1],
            act_layer(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            build_norm_layer(norm_layer, 64)[1],
            act_layer(),
            nn.Conv2d(64, channel, 1),
            build_norm_layer(norm_layer, channel)[1]
        )

    def forward(self, x):
        return self.block(x)


class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]]).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        return self.conv_extra(x + scharr_edge)


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer):
        super().__init__()
        kernel = self.gaussian_kernel(size, sigma)
        kernel = nn.Parameter(kernel, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, padding=size // 2, groups=dim, bias=False)
        self.gaussian.weight.data = kernel.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()

    def forward(self, x):
        gaussian = self.gaussian(x)
        return self.act(self.norm(gaussian))

    def gaussian_kernel(self, size, sigma):
        return torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)


class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        return self.norm(c + att * wei)


class EGA_Module(nn.Module):
    def __init__(self, dim, stage, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()
        self.stage = stage
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(p=drop_path)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.lfea = LFEA(dim, norm_layer, act_layer)
        if stage == 0:
            self.edge_extract = Scharr(dim, norm_layer, act_layer)
        else:
            self.edge_extract = Gaussian(dim, 5, 1.0, norm_layer, act_layer)

    def forward(self, x):
        att = self.edge_extract(x)
        x_att = self.lfea(x, att)
        return x + self.norm(self.drop_path(self.mlp(x_att)))

if __name__ == '__main__':


    # 示例输入张量：batch_size=4, channels=64, height=64, width=64
    input = torch.randn(4, 64, 64, 64)
    print(input.size())

    # 实例化 EGA 模块（使用 stage=0 即边缘增强分支）
    block = EGA_Module(
        dim=64,
        stage=0,
        mlp_ratio=2.0,
        drop_path=0.1,
        norm_layer=dict(type='BN', requires_grad=True),
        act_layer=nn.ReLU
    )

    # 通过模块处理输入
    output = block(input)
    print(output.size())
