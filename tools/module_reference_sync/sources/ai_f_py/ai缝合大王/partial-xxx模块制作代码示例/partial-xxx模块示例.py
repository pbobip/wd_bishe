import torch
from thop import profile
from torch import nn

from MDTA import Attention
from MSLA import LiteMLA


# 此为 创新思维开篇 中的Partial-xxx模块制作代码示例
class partialXXX模块示例(nn.Module):
    def __init__(self, dim, n_div=4):  # n_div 通道系数，如设置为2，则只对1/2的通道数进行操作，设置为4，只对1/4的通道数进行操作
        super(partialXXX模块示例, self).__init__()
        self.dim_block = dim // n_div  # 需要进行处理（如卷积、注意力机制）的通道数
        self.dim_identity = dim - self.dim_block  # 不进行任何操作的通道数
        # scales: 单尺度:(5,); 多尺度:(3,5)
        # self.block = LiteMLA(in_channels=self.dim_block, out_channels=self.dim_block, scales=(3,5))
        self.block = Attention(self.dim_block)
        # 我们在这里，可以直接引用别的模块，Github开源即插即用模块，https://github.com/ai-dawang/PlugNPlay-Modules
        # 卷积、注意力机制、各种模块等都可以采用这个“部分”思想，相当于这一个思想，就可以套在上百个模块中。
        # 我们可以命名这个模块为partialXXX模块，我们可以再此模块基础上，再进行同其他模块的缝合
        self.project = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_block, self.dim_identity], dim=1)  # 进行通道划分，x1进行操作，x2不进行任何操作
        x1 = self.block(x1)
        x = torch.cat((x1, x2), 1)  # x1 x2 进行通道合并
        x = self.project(x)  # 通过个1x1卷积处理一下
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 只对一部分通道进行处理，效果可能相当，可能不如对全部通道处理性能好
    # 不过使用一部分，模型可以有更小的参数和flops，，，我们可以加大模块的堆叠个数，或者加大通道数，，，这样同参数下，效果会好一点。
    block = partialXXX模块示例(dim=64, n_div=4).cuda()
    # block = Attention(dim=64).cuda()
    x = torch.rand(3, 64, 128, 128).cuda()
    output = block(x)
    flops, params = profile(block, (x,))
    print(flops / 1e9)
    print(params)
    print(x.size(), output.size())
