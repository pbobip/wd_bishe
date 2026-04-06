import torch
import torch.nn as nn

'''
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
论文题目：Dual Residual Attention Network for Image Denoising （2024）
'''

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        # 记录输出通道数
        self.out_channels = out_planes
        groups = 1  # 默认采用标准卷积（无分组）
        # 初始化二维卷积层，指定卷积核大小、填充和分组数
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        # 使用ReLU激活以增加非线性表达
        self.relu = nn.ReLU()

    def forward(self, x):
        # 依次执行卷积和ReLU激活操作
        x = self.conv(x)
        x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        # 在通道维度上提取最大值和均值，并在新的通道维度上连接这两个结果
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5  # 设定空间卷积的核尺寸
        self.compress = ChannelPool()  # 通过池化压缩通道信息
        # 利用Basic模块执行空间卷积处理，输出单通道注意力图
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # ai缝合大王

    def forward(self, x):
        # 压缩输入的通道信息为两个统计量
        x_compress = self.compress(x)
        # 使用卷积模块计算空间注意力图
        x_out = self.spatial(x_compress)
        # 用Sigmoid将卷积输出映射到[0,1]，作为通道权重
        scale = torch.sigmoid(x_out)
        # 对原始输入进行加权，突出重要区域
        return x * scale

class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3  # 使用3x3卷积
        stride = 1
        padding = 1
        layers = []
        # 第一层卷积加ReLU，初步提取特征
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        # 第二层卷积进一步提取信息
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # 将上述卷积层整合为一个序列
        self.res = nn.Sequential(*layers)
        # 初始化空间注意力模块以精细调制特征
        self.sab = SAB()

    def forward(self, x):
        # 第一次残差连接：将输入与卷积结果相加
        x1 = x + self.res(x)
        # 第二次残差连接：在x1的基础上再次融合卷积输出
        x2 = x1 + self.res(x1)
        # 第三次残差连接：累加卷积处理结果进一步强化特征
        x3 = x2 + self.res(x2)
        # 融合初级和高级信息：将x1与x3相加
        x3_1 = x1 + x3
        # 第四次残差连接：对融合结果再次进行卷积处理并加上原融合结果
        x4 = x3_1 + self.res(x3_1)
        # 将初始输入与处理后的特征相加，形成全局残差连接
        x4_1 = x + x4
        # 通过空间注意力模块对特征进行细粒度的调制
        x5 = self.sab(x4_1)
        # 最终将注意力调整后的特征与原始输入融合输出
        x5_1 = x + x5
        return x5_1

if __name__ == '__main__':
    # 生成一个批量大小为1，64通道，256x256分辨率的随机输入张量
    input_tensor = torch.randn(1, 64, 256, 256)
    # 实例化RAB模块
    rab = RAB(in_channels=64, out_channels=64, bias=True)
    # 通过RAB模块获取输出特征
    rab_output = rab(input_tensor)
    print("RAB 输入维度:", input_tensor.shape)
    print("RAB 输出维度:", rab_output.shape)
    # ai缝合大王
