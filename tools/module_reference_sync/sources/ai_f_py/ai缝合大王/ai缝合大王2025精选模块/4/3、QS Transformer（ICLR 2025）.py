import torch.nn.functional as F
import torch
import torch.nn as nn
from quan_w import Conv2dLSQ

"""
    论文来源：https://arxiv.org/pdf/2501.13492
    论文标题：QUANTIZED SPIKE-DRIVEN TRANSFORMER (ICLR 2025)
"""

# 批归一化（Batch Normalization）和填充（Padding）自定义层
class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,           # 填充像素数
            num_features,         # 通道数（特征数）
            eps=1e-5,             # 防止除零的小值
            momentum=0.1,         # 动量参数
            affine=True,          # 是否使用可学习的缩放和偏移参数
            track_running_stats=True,  # 是否跟踪运行时的均值和方差
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels  # 记录填充像素数

    def forward(self, input):
        output = self.bn(input)  # 执行批归一化
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

# 可重参数化卷积模块
class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        conv1x1 = Conv2dLSQ(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            Conv2dLSQ(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            Conv2dLSQ(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

# 受限 ReLU 激活函数，范围 [0, thre]
class ReLUX(nn.Module):
    def __init__(self, thre=8):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(thre=4)

# 多脉冲激活函数
class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None

# 多脉冲激活模块
class Multispike(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 4

class Multispike_att(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 2

# 多脉冲注意力机制
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须可以被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        self.head_lif = Multispike()
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()
        self.attn_lif = Multispike_att()
        self.proj_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        N = H * W
        x = self.head_lif(x)
        q = self.q_lif(self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)).flatten(3)
        k = self.k_lif(self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)).flatten(3)
        v = self.v_lif(self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)).flatten(3)
        x = (q @ (k.transpose(-2, -1) @ v)) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x.squeeze(0)

if __name__ == "__main__":
    input_tensor = torch.randn(4, 64, 32, 32)
    model = MS_Attention_RepConv_qkv_id(dim=64, num_heads=8)
    output_tensor = model(input_tensor)
    print("输出张量形状：", output_tensor.shape)
