import torch
import torch.nn as nn
#论文：LSNet: See Large, Focus Small
#论文地址：https://arxiv.org/pdf/2503.23135
# ===== Conv2d + BN =====
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


# ===== LKP: Lightweight Kernel Prediction =====
class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


# ===== PyTorch 实现的 SKA 替代版（不依赖 Triton）=====
class SKA(nn.Module):
    def forward(self, x, w):
        """
        x: [B, C, H, W]
        w: [B, G, K*K, H, W]
        G = C // groups
        """
        B, C, H, W = x.shape
        G = w.shape[1]
        K = int(w.shape[2] ** 0.5)
        pad = K // 2
        out = torch.zeros_like(x)

        x_unfold = torch.nn.functional.unfold(x, kernel_size=K, padding=pad)  # [B, C*K*K, H*W]
        x_unfold = x_unfold.view(B, G, C // G, K * K, H, W)  # [B, G, C//G, K*K, H, W]

        w = w.view(B, G, 1, K * K, H, W)  # [B, G, 1, K*K, H, W]
        out_group = (x_unfold * w).sum(dim=3)  # [B, G, C//G, H, W]
        out = out_group.view(B, C, H, W)
        return out


# ===== 最终模块：LSConv =====
class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


# ===== ✅ 实例测试 =====
if __name__ == '__main__':
    x = torch.randn(4, 64, 32, 32)  # CPU上可跑，如需GPU可加 .cuda()
    model = LSConv(64)
    print("Input:", x.shape)
    y = model(x)
    print("Output:", y.shape)
