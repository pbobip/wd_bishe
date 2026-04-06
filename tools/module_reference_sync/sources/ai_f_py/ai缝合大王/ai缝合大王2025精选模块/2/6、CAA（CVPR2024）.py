import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

"""
论文链接：https://arxiv.org/pdf/2403.06258
论文题目：Poly Kernel Inception Network for Remote Sensing Detection (CVPR 2024)
"""

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,      # 输入特征图通道数
            out_channels: int,     # 输出特征图通道数
            kernel_size: int,      # 卷积核尺寸
            stride: int = 1,       # 步长，默认为1
            padding: int = 0,      # 填充，默认为0
            groups: int = 1,       # 分组卷积数，默认为1
            norm_cfg: Optional[dict] = None,  # 归一化配置
            act_cfg: Optional[dict] = None):  # 激活函数配置
        super().__init__()
        layers = []
        # 构建卷积层，自动设置填充为 (kernel_size-1)//2 当padding未指定时
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding=(kernel_size - 1) // 2 if padding is None else padding,
                                groups=groups, bias=(norm_cfg is None)))
        # 添加归一化层（如有配置）
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # 添加激活层（如有配置）
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)
        # ai缝合大王

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1),
                                  eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention 模块，用于捕捉局部和全局语义信息"""
    def __init__(
            self,
            channels: int,  # 输入通道数
            h_kernel_size: int = 11,  # 水平卷积核尺寸，默认为11
            v_kernel_size: int = 11,  # 垂直卷积核尺寸，默认为11
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置
            act_cfg: Optional[dict] = dict(type='SiLU')):  # 激活函数配置
        super().__init__()
        # 使用7x7平均池化，保持空间信息
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=3)
        # 先通过1x1卷积调整特征分布
        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 对输入进行水平方向卷积操作，提取局部水平方向信息；采用分组卷积确保每个通道独立处理
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 padding=(0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 对水平方向输出再进行垂直方向卷积处理
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 padding=(v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 通过1x1卷积整合卷积信息
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 使用Sigmoid函数生成最终注意力因子
        self.act = nn.Sigmoid()
        # ai缝合大王

    def forward(self, x):
        avg = self.conv1(self.avg_pool(x))
        # 分别在水平和垂直方向上提取特征
        h_conv = self.h_conv(avg)
        v_conv = self.v_conv(h_conv)
        attn_factor = self.act(self.conv2(v_conv))
        return attn_factor

if __name__ == "__main__":
    # 随机生成输入张量，形状为 (1, 64, 128, 128)
    input_tensor = torch.randn(1, 64, 128, 128)
    caa = CAA(64)  # 实例化CAA模块
    output_tensor = caa(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    # ai缝合大王
