import torch
from thop import profile
from torch import nn
from MDTA import Attention
from SFB import SFB
from 多尺度线性注意力机制 import LiteMLA


class 自制模块例子(nn.Module):
    def __init__(self, dim, scale=4):
        super(自制模块例子, self).__init__()

        dim1 = dim // scale
        dim2 = int(dim1 * (scale ** 2))

        self.down = nn.Sequential(
            nn.Conv2d(dim, dim1, 1),
            nn.Conv2d(dim1, dim1, scale, scale, groups=dim1)
        )

        # self.att = Attention(dim1)
        self.att = LiteMLA(in_channels=dim1, out_channels=dim1, scales=(3, 5))  # scales: 单尺度:(5,); 多尺度:(3,5)
        self.sfb = SFB(dim1)

        self.up = nn.Sequential(
            nn.Conv2d(dim1, dim2, 1, groups=dim1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim1, dim, 1)
        )

    def forward(self, x):
        x = self.down(x)  # 下采样
        x = self.att(x)  # 注意力模块
        x = self.sfb(x)  # 频域模块
        x = self.up(x)  # 上采样模块
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = 自制模块例子(dim=64, scale=4).cuda()  # scale下采样倍数
    # block = Attention(dim=64).cuda()
    x = torch.rand(3, 64, 128, 128).cuda()
    output = block(x)
    flops, params = profile(block, (x,))
    print(flops / 1e9)
    print(params)
    print(x.size(), output.size())
