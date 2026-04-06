# Code Implementation of the MaIR Model
import torch.nn as nn
import torch


# MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration
# 论文：https://arxiv.org/pdf/2412.20066v2
# Github:https://github.com/XLearning-SCU/2025-CVPR-MaIR

class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, group=4, act_layer=nn.GELU,
                 input_resolution=(64, 64)):
        super().__init__()
        self.group = group
        self.input_resolution = input_resolution
        self.in_features = in_features
        self.out_features = out_features

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def channel_rearrange(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        res = x
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)
        # return x
        return x * res


# 输入 B C H W,  输出B C H W
if __name__ == '__main__':
    block = ShuffleAttn(in_features=64, out_features=64).cuda()
    x = torch.rand(3, 64, 128, 128).cuda()
    output = block(x)
    print(x.size(), output.size())
