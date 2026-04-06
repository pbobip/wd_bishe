import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from timm.models.layers import to_2tuple


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


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=30, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        # print(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x, y):
        B, H, W, _ = x.shape
        # print(x.shape,y.shape)
        routeing = self.reweight(y.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        # print(x.shape)
        x = self.act1(x)
        x = x.to(torch.float32)
        # print(x.shape)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        # print(complex_weights.shape,weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)

        # plt.close(fig)
        return x


class FDSM(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv_rgb = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False)
        self.conv_nir = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False)
        self.softmax = nn.SiLU()
        self.pool = nn.Sequential(nn.Conv2d(c, c, 1, 1, 0, bias=False),
                                  nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
                                  LayerNorm(c, LayerNorm_type='WithBias'),
                                  nn.PReLU())
        self.fc_ab = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, 1, 0, bias=False))
        self.dynamic_rgb = DynamicFilter(c)
        self.dynamic_nir = DynamicFilter(c)

    def forward(self, rgb, nir):
        feat_1 = self.conv_rgb(rgb)
        feat_2 = self.conv_nir(nir)
        feat_sum = feat_1 + feat_2
        s = self.pool(feat_sum)
        z = s
        ab = self.fc_ab(z)
        B, C, H, W = ab.shape
        ab = ab.view(B, 2, C // 2, H, W)
        ab = self.softmax(ab)
        a = ab[:, 0, ...]
        b = ab[:, 1, ...]
        feat_1 = self.dynamic_rgb(feat_1.permute(0, 2, 3, 1), a.permute(0, 2, 3, 1))
        feat_2 = self.dynamic_nir(feat_2.permute(0, 2, 3, 1), b.permute(0, 2, 3, 1))

        return feat_1.permute(0, 3, 1, 2), feat_2.permute(0, 3, 1, 2)


# 输入 B C H W,  输出B C H W
if __name__ == '__main__':
    block = FDSM(c=64).cuda()
    x_0 = torch.randn((3, 64, 128, 128)).cuda()
    x_1 = torch.randn((3, 64, 128, 128)).cuda()
    output_0, output_1 = block(x_0, x_1)
    print(x_0.size(), x_1.size(), output_0.size(), output_1.size())
