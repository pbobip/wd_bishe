import torch  # 引入PyTorch库
import torch.nn as nn  # 引入PyTorch神经网络模块

"""
论文链接：https://openreview.net/pdf?id=XhYWgjqCrV
论文名称：MOGANET: MULTI-ORDER GATED AGGREGATION NETWORK（ICLR 2024）
"""

class ElementScale(nn.Module):

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()  # 调用基类的构造函数

        # 生成一个可训练的缩放参数self.scale，用于在前向传播中对输入逐元素放大或缩小。
        # 例如，当embed_dims为64且初始值为0.5时，self.scale会被初始化为尺寸为(1, 64, 1, 1)的张量，所有元素为0.5。
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),  # 初始化缩放因子
            requires_grad=requires_grad  # 指定该参数是否需要梯度更新
        )
        # ai缝合大王

    def forward(self, x):
        return x * self.scale  # 返回输入与缩放参数逐元素相乘后的结果

class ChannelAggregationFFN(nn.Module):
    """实现带有通道聚合的前馈网络(FFN)。

    参数说明:
        embed_dims (int): 输入特征的维度。
        ffn_ratio: 用于扩展隐藏层通道数的倍率。
        kernel_size (int): 深度卷积核的尺寸，默认值为3。
        ffn_drop (float, optional): 前馈网络中Dropout的概率，默认值为0.0。
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 kernel_size=3,
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()  # 调用父类的初始化方法

        self.embed_dims = embed_dims  # 保存输入特征维度

        # 计算前馈网络隐藏层的通道数
        feedforward_channels = int(embed_dims * ffn_ratio)
        self.feedforward_channels = feedforward_channels  # 记录隐藏层维度

        # 构造第一个1x1卷积层，用于将输入特征投影到更高维空间
        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,  # 输入通道数
            out_channels=self.feedforward_channels,  # 扩展后的输出通道数
            kernel_size=1  # 卷积核尺寸为1
        )

        # 构建深度卷积层，实现空间特征的局部提取
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,  # 输入通道数
            out_channels=self.feedforward_channels,  # 输出通道数保持不变
            kernel_size=kernel_size,  # 使用给定的卷积核尺寸
            stride=1,  # 设置步长为1
            padding=kernel_size // 2,  # 使用适当的填充保持尺寸不变
            bias=True,  # 包含偏置项
            groups=self.feedforward_channels  # 分组卷积实现深度卷积
        )

        self.act = nn.GELU()  # 使用GELU激活函数引入非线性

        # 构造第二个1x1卷积层，用于将特征映射还原到原始维度
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,  # 输入通道数
            out_channels=embed_dims,  # 输出通道数恢复为原始维度
            kernel_size=1  # 卷积核尺寸为1
        )

        self.drop = nn.Dropout(ffn_drop)  # 定义Dropout层以实现正则化

        # 构造降维卷积，将多通道特征聚合为单通道表示
        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # 多通道输入
            out_channels=1,  # 降维为单通道输出
            kernel_size=1  # 使用1x1卷积实现降维
        )

        # 创建ElementScale实例，用于调节降维后特征的重要性
        self.sigma = ElementScale(
            self.feedforward_channels,  # 传入隐藏层通道数
            init_value=1e-5,  # 设定初始缩放值
            requires_grad=True  # 参数在训练中可更新
        )

        self.decompose_act = nn.GELU()  # 对降维输出应用GELU激活函数
        # ai缝合大王

    # 关键操作：特征分解
    def feat_decompose(self, x):

        Temp = self.decompose(x)            # 将输入x经过1x1卷积降维，从形状[B, C, H, W]变为[B, 1, H, W]，提取主要特征
        Temp = self.decompose_act(Temp)      # 对降维结果应用GELU激活，增强非线性表达
        Temp = x - Temp                    # 从原始特征x中减去主要特征Temp，保留局部细节信息
        Temp = self.sigma(Temp)              # 使用可学习的缩放因子对差异特征进行调节
        x = x + Temp                         # 将调节后的特征加回原始输入，实现信息融合
        return x

    def forward(self, x):
        # 首次投影：使用1x1卷积将输入映射到更高维空间
        x = self.fc1(x)  # 应用第一个卷积层进行投影
        x = self.dwconv(x)  # 执行深度卷积提取局部特征
        x = self.act(x)  # 应用GELU激活函数
        x = self.drop(x)  # 施加Dropout正则化

        # 进行特征分解，提取并调节关键特征
        x = self.feat_decompose(x)  # 对投影后的特征进行分解操作

        # 二次投影：利用1x1卷积将特征映射还原至原始维度
        x = self.fc2(x)  # 使用第二个卷积层恢复通道数
        x = self.drop(x)  # 再次施加Dropout正则化
        return x

if __name__ == '__main__':
    input = torch.randn(1, 64, 32, 32)  # 生成随机输入张量，尺寸为(1, 64, 32, 32)

    CA = ChannelAggregationFFN(64)  # 创建ChannelAggregationFFN实例，特征维度设为64

    output = CA(input)  # 将输入数据传入模型进行前向传播

    print(' CA_input_size:', input.size())  # 打印输入张量的尺寸
    # ai缝合大王
    print(' CA_output_size:', output.size())  # 打印输出张量的尺寸
