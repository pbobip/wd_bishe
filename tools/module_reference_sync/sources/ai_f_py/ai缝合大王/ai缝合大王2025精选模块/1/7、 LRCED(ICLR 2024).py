import torch  # 导入 PyTorch 库，用于张量操作和自动求导
import torch.nn as nn  # 导入 PyTorch 神经网络模块，构建各种层
from timm.models.layers import trunc_normal_, DropPath, to_2tuple  # 从 timm 库引入一些辅助函数
act_layer = nn.ReLU  # 设置默认激活函数为 ReLU
ls_init_value = 1e-6  # 定义微小的初始缩放因子值
# ai缝合大王

'''
论文链接：https://arxiv.org/abs/2302.06052 
论文标题：CEDNET: A CASCADE ENCODER-DECODER NETWORK FOR DENSE PREDICTION (ICLR 2024)
'''

class LRCED(nn.Module):
    def __init__(self, dim, drop_path=0., dilation=3, **kwargs):
        """
        构造 LRCED 模块，该模块结合了深度可分离卷积和逐点全连接操作，
        以实现编码器-解码器结构中的密集预测。
        """
        super().__init__()  # 调用父类构造函数初始化网络

        # 第一组标准深度卷积模块：使用 7x7 卷积、批归一化和激活函数
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),
            nn.BatchNorm2d(dim),
            act_layer()
        )

        # 第二组扩张深度卷积模块：采用扩张率为 dilation 的 7x7 卷积，扩大感受野
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            nn.BatchNorm2d(dim),
            act_layer()
        )

        # 逐点卷积部分：通过全连接层（1x1 卷积的等价实现）扩展通道维度后再恢复回原维度
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()  # 应用激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # 定义一个可学习的缩放参数，用于细调全连接层输出；当 ls_init_value 大于 0 时启用
        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        # 随机丢弃路径（DropPath）机制，用于增强网络鲁棒性；当 drop_path 为 0 时，恒等映射被使用
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # 记录原始输入以便残差连接使用

        # 分别经过两个深度卷积模块，并在每一步中添加输入以实现残差学习
        x = self.dwconv1(x) + x
        x = self.dwconv2(x) + x

        # 调整张量维度，以便全连接层对最后一个维度进行处理
        x = x.permute(0, 2, 3, 1)  # 从 (N, C, H, W) 转换为 (N, H, W, C)
        x = self.pwconv1(x)  # 扩展特征维度
        x = self.act(x)      # 应用激活函数，增加非线性表达  # ai缝合大王
        x = self.pwconv2(x)  # 恢复到原始维度
        if self.gamma is not None:
            x = self.gamma * x  # 对输出进行缩放

        # 将张量维度还原回 (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # 将经过全连接层处理的结果与原始输入相加，并应用 DropPath 随机丢弃机制
        x = input + self.drop_path(x)
        return x

if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)  # 随机生成形状为 (1, 64, 32, 32) 的输入张量
    model = LRCED(64)  # 实例化 LRCED 模块，设定特征通道数为 64
    output = model(input)  # 前向传播计算输出
    print('input_size:', input.size())  # 打印输入张量尺寸
    print('output_size:', output.size())  # 打印输出张量尺寸
    # ai缝合大王
