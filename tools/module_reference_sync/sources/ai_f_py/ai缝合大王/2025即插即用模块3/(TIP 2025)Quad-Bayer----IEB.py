import torch
import torch.nn as nn
import torch.nn.functional as F
#论文地址：https://arxiv.org/html/2412.07256v1
#论文：Modeling Dual-Exposure Quad-Bayer Patterns for Joint Denoising and Deblurring
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError("Unsupported padding type: {}".format(pad_type))

        # Initialize normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        # Initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        # Initialize convolution
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class IEB(nn.Module):
    """
    Input Enhancement Block (IEB)
    输入：多尺度下采样特征（avgpool down2/4/8 + pixel shuffle down4）
    输出：融合后的特征增强图
    """
    def __init__(self, in_channels=1, start_channels=32, latent_dim=1, pad='zero', norm='none'):
        super(IEB, self).__init__()

        self.x_avgpooldown2_recover_layer = Conv2dLayer(in_channels, in_channels, 3, 2, 1, pad_type=pad, norm=norm)
        self.x_avgpooldown8_recover_layer = Conv2dLayer(in_channels, in_channels, 3, 1, 1, pad_type=pad, norm=norm)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.combine_layer = Conv2dLayer(int(in_channels * (3 + 16)), start_channels, 3, 1, 1, pad_type=pad, norm=norm)
        self.fc_diff_1 = nn.Linear(start_channels, start_channels // latent_dim)
        self.fc_diff_2 = nn.Linear(start_channels // latent_dim, start_channels)

        self.conv_x = Conv2dLayer(int(in_channels * 16), start_channels, 3, 1, 1, pad_type=pad, norm=norm)
        self.final = Conv2dLayer(start_channels * 2, start_channels, 3, 1, 1, pad_type=pad, norm=norm)

    def forward(self, x_avgpooldown2, x_avgpooldown4, x_avgpooldown8, x_psdown4):
        x_avgpooldown2 = self.x_avgpooldown2_recover_layer(x_avgpooldown2)  # [B, C, H, W]
        x_avgpooldown8 = self.x_avgpooldown8_recover_layer(x_avgpooldown8)
        x_avgpooldown8 = F.interpolate(x_avgpooldown8, scale_factor=2, mode='bilinear', align_corners=False)

        # 拼接维度对齐后进行融合
        fea_comb_x = torch.cat((x_avgpooldown2, x_avgpooldown4, x_avgpooldown8, x_psdown4), dim=1)
        fea_comb_x = self.combine_layer(fea_comb_x)

        residual_fea = fea_comb_x
        fea = self.gap(fea_comb_x).view(fea_comb_x.size(0), -1)
        fea = self.fc_diff_1(fea)
        fea = self.fc_diff_2(fea)
        fea = self.sigmoid(fea).view(fea.size(0), fea.size(1), 1, 1)
        fea_comb_x = residual_fea * fea

        x_self = self.conv_x(x_psdown4)
        out = self.final(torch.cat((x_self, fea_comb_x), dim=1))
        return out


if __name__ == '__main__':
    # 模拟图像下采样后的多尺度输入
    batch_size = 2
    in_channels = 1
    h, w = 64, 64  # 中心尺度分辨率

    # 构造输入
    x_avgpooldown2 = torch.randn(batch_size, in_channels, h * 2, w * 2)
    x_avgpooldown4 = torch.randn(batch_size, in_channels, h, w)
    x_avgpooldown8 = torch.randn(batch_size, in_channels, h // 2, w // 2)
    x_psdown4 = torch.randn(batch_size, in_channels * 16, h, w)  # PixelUnShuffle 特征图

    # 实例化 IEB 模块
    model = IEB(in_channels=in_channels, start_channels=32)

    # 前向传播
    out = model(x_avgpooldown2, x_avgpooldown4, x_avgpooldown8, x_psdown4)
    print("Output shape:", out.shape)  # 应为 [B, start_channels, H, W]
