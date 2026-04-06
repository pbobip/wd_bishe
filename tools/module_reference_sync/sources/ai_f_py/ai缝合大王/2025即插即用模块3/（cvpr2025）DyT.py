import torch
import torch.nn as nn


# Transformers without Normalization
# 论文：https://arxiv.org/pdf/2503.10622
# Github地址：https://github.com/jiachenzhu/DyT

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


# 输入 B H W C,  输出B H W C     可用DyT替换归一化层
if __name__ == '__main__':
    block = DyT(num_features=64).cuda()
    x = torch.rand(3, 128, 128, 64).cuda()
    output = block(x)
    print(x.size(), output.size())
