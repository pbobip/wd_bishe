import torch
import torch.nn as nn

'''
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
论文题目：Dual Residual Attention Network for Image Denoising （2024）
'''

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        # 初始化Basic模块，构建一个卷积层及其后续ReLU激活
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1  # 默认不使用分组卷积
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()  # 使用ReLU激活函数

    def forward(self, x):
        # 先进行卷积，再激活
        x = self.conv(x)
        x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def __init__(self):
        # 初始化ChannelPool模块，用于聚合通道的最大值和均值
        super(ChannelPool, self).__init__()

    def forward(self, x):
        # 在通道维度上分别计算最大值和平均值，并沿新维度拼接
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5  # 设置卷积核尺寸为5
        self.compress = ChannelPool()  # 通道池化，用于压缩输入信息
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)  # 空间注意力中的卷积处理
        # ai缝合大王

    def forward(self, x):
        # 压缩通道特征信息
        x_compress = self.compress(x)
        # 通过基本卷积块获得空间注意力图
        x_out = self.spatial(x_compress)
        # 利用Sigmoid函数获得归一化的注意力权重，并加权输入
        scale = torch.sigmoid(x_out)
        return x * scale

class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3  # 卷积核尺寸
        stride = 1
        padding = 1
        layers = []
        # 第一组卷积和激活层：提取初级特征
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        # 第二个卷积层：进一步提取特征
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)  # 将卷积层堆叠成序列
        self.sab = SAB()  # 初始化空间注意力模块

    def forward(self, x):
        # 第一次残差连接：输入加上第一次卷积结果
        x1 = x + self.res(x)
        # 第二次残差连接：在x1上重复卷积并与x1相加
        x2 = x1 + self.res(x1)
        # 第三次残差连接：在x2上再次卷积处理后加上x2
        x3 = x2 + self.res(x2)
        # 跳跃连接：将x1与x3叠加以融合低级与高级信息
        x3_1 = x1 + x3
        # 第四次残差连接：对x3_1应用卷积后与x3_1相加
        x4 = x3_1 + self.res(x3_1)
        # 与原始输入融合
        x4_1 = x + x4
        # 使用空间注意力模块细化特征响应
        x5 = self.sab(x4_1)
        # 最终输出为原始输入与注意力调制结果的融合
        x5_1 = x + x5
        return x5_1

if __name__ == '__main__':
    # 构造随机输入，尺寸为 (1, 64, 256, 256)
    input_tensor = torch.randn(1, 64, 256, 256)
    # 初始化 RAB 模块
    rab = RAB(in_channels=64, out_channels=64, bias=True)
    # 计算 RAB 模块的输出
    rab_output = rab(input_tensor)
    print("RAB 输入维度:", input_tensor.shape)
    print("RAB 输出维度:", rab_output.shape)
    # ai缝合大王
