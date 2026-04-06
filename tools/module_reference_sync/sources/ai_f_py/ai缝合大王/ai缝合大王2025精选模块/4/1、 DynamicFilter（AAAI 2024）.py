import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple
"""
    论文来源：https://ojs.aaai.org/index.php/AAAI/article/download/29457/30746
    论文标题：FFT-Based Dynamic Token Mixer for Vision（AAAI 2024）
"""

class StarReLU(nn.Module):
    """
    StarReLU 激活函数：s * relu(x) ** 2 + b
    其中 s 和 b 是可学习参数。
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()  # 调用父类 nn.Module 的构造函数
        self.inplace = inplace  # 是否执行原地计算
        self.relu = nn.ReLU(inplace=inplace)  # 定义 ReLU 激活层
        # 定义可训练的缩放参数 s，默认值 scale_value，是否参与梯度更新由 scale_learnable 决定
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                              requires_grad=scale_learnable)
        # 定义可训练的偏置参数 b，默认值 bias_value，是否参与梯度更新由 bias_learnable 决定
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                             requires_grad=bias_learnable)

    def forward(self, x):
        """
        前向传播，计算 StarReLU 激活结果。
        """
        return self.scale * self.relu(x) ** 2 + self.bias  # 应用 StarReLU 计算公式

# 定义 MLP（多层感知机）模块，常用于 MetaFormer 系列模型
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim  # 输入特征维度
        out_features = out_features or in_features  # 默认输出特征维度等于输入特征维度
        hidden_features = int(mlp_ratio * in_features)  # 计算隐藏层维度
        drop_probs = to_2tuple(drop)  # 将 dropout 概率转换为元组

        # 第一个全连接层，将输入映射到隐藏层维度
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()  # 激活函数，默认使用 StarReLU
        self.drop1 = nn.Dropout(drop_probs[0])  # 第一层 dropout
        # 第二个全连接层，将隐藏层映射回输出维度
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])  # 第二层 dropout

    def forward(self, x):
        """
        前向传播，依次经过全连接层、激活函数、dropout。
        """
        x = self.fc1(x)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop1(x)  # 第一个 dropout 层
        x = self.fc2(x)  # 第二个全连接层
        x = self.drop2(x)  # 第二个 dropout 层
        return x

# 定义动态滤波器模块，用于处理多维特征数据
class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)  # 将 size 解析为元组（高度，宽度）
        self.size = size[0]  # 高度
        self.filter_size = size[1] // 2 + 1  # FFT 计算所需的滤波器大小
        self.num_filters = num_filters  # 滤波器数量
        self.med_channels = int(expansion_ratio * dim)  # 中间通道数

        # 第一个逐点卷积层，用于调整通道数
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()  # 第一个激活函数层
        # 计算权重的 MLP 层
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        # 初始化复数滤波器权重
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()  # 第二个激活函数层
        # 第二个逐点卷积层
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        """
        前向传播，实现动态滤波计算。
        """
        x = x.permute(0, 2, 3, 1)  # 调整维度顺序（B, H, W, C）
        B, H, W, _ = x.shape  # 获取输入的形状

        # 计算路由权重，基于输入的空间平均值
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)

        x = self.pwconv1(x)  # 第一层逐点卷积
        x = self.act1(x)  # 激活函数
        x = x.to(torch.float32)  # 确保数据类型为 float32

        # 进行二维 FFT 变换
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # 计算滤波器加权
        complex_weights = torch.view_as_complex(self.complex_weights)  # 转换为复数权重
        routeing = routeing.to(torch.complex64)  # 转换路由权重为复数
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)  # 计算加权滤波器
        weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x = x * weight  # 乘以滤波器权重

        # 进行逆 FFT 变换，恢复到原始空间
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)  # 第二个激活函数层
        x = self.pwconv2(x)  # 第二个逐点卷积层

        return x.permute(0, 3, 1, 2)  # 还原维度顺序（B, C, H, W）

if __name__ == '__main__':
    block = DynamicFilter(32, size=64)  # 定义 DynamicFilter 模块
    input = torch.rand(1, 32, 64, 64)  # 创建随机输入张量
    output = block(input)  # 计算输出
    print(input.size())  # 打印输入形状
    print(output.size())  # 打印输出形状