import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import torch_dct as dct
from einops import rearrange
import math


# Complementary Advantages: Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising
# 论文：https://arxiv.org/pdf/2412.16645v1
# Github:https://github.com/11679-hub/11679

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class FEFM(nn.Module):
    def __init__(self, dim, bias, depth):
        super(FEFM, self).__init__()
        self.num_heads = dim // 16
        # print(depth)
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.to_hidden = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_hidden_nir = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw_nir = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                          bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.special = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_middle = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.pool1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))
        self.pool2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x, nir):
        b, c, h, w = x.shape
        hidden = self.to_hidden(x)
        nir_hidden = self.to_hidden_nir(nir)
        q = self.to_hidden_dw(hidden)
        k, v = self.to_hidden_dw_nir(nir_hidden).chunk(2, dim=1)
        # print(q.shape,k.shape)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        q_fft = dct.dct_2d(q_patch.float())
        k_fft = dct.dct_2d(k_patch.float())
        # print(q_fft.shape)
        out1 = q_fft * k_fft
        # print(out1.shape)
        q_fft = rearrange(q_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        k_fft = rearrange(k_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        out1 = rearrange(out1, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        q_fft = torch.nn.functional.normalize(q_fft, dim=-1)
        k_fft = torch.nn.functional.normalize(k_fft, dim=-1)
        attn = (q_fft @ k_fft.transpose(-2, -1)) * self.temperature
        # print(attn.shape)
        # attn=attn.real
        # print(attn.shape)
        attn = attn.softmax(dim=-1)
        out = (attn @ out1)
        # print(out.shape)
        out = rearrange(out, 'b head c (h w patch1 patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                        h=h // self.patch_size, w=w // self.patch_size, patch1=self.patch_size,
                        patch2=self.patch_size)

        out = dct.idct_2d(out)
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        out = self.project_middle(out)
        # out = self.norm(out)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        # print(lambda_full)

        output = self.pool1(q * out) + self.pool2(v - lambda_full * (v * out))
        output = self.project_out(output)

        return output


# 输入 B C H W,  输出B C H W
if __name__ == '__main__':
    block = FEFM(dim=64, bias=False, depth=1).cuda()
    x_0 = torch.randn((3, 64, 128, 128)).cuda()
    x_1 = torch.randn((3, 64, 128, 128)).cuda()
    output = block(x_0, x_1)
    print(x_0.size(), x_1.size(), output.size())


