import torch
import torch.nn as nn
import torch.nn.functional as F
#论文地址：Modeling Dual-Exposure Quad-Bayer Patterns for Joint Denoising and Deblurring
#论文：Frequency Dynamic Convolution for Dense Image Prediction
class FDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 kernel_num=4, param_ratio=1, param_reduction=1.0, kernel_temp=1.0):
        super(FDConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.kernel_temp = kernel_temp

        # 初始化 FFT 权重
        d1, d2 = out_channels * kernel_size, in_channels * kernel_size
        self.freq_indices, _ = self.get_fft2freq(d1, d2, use_rfft=True)

        num_freq = self.freq_indices.shape[1]
        if self.param_reduction < 1:
            num_freq = int(num_freq * self.param_reduction)
            self.freq_indices = self.freq_indices[:, :num_freq]

        self.dft_weight = nn.Parameter(
            torch.randn(self.kernel_num, num_freq, 2) * 1e-6
        )  # [kernel_num, freq, 2]

    def get_fft2freq(self, d1, d2, use_rfft=False):
        freq_h = torch.fft.fftfreq(d1)
        freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
        freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
        dist = torch.norm(freq_hw, dim=-1)
        sorted_dist, indices = torch.sort(dist.view(-1))
        if use_rfft:
            d2 = d2 // 2 + 1
        sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
        return sorted_coords.permute(1, 0), freq_hw

    def forward(self, x):
        b, c, h, w = x.shape
        d1 = self.out_channels * self.kernel_size
        d2 = self.in_channels * self.kernel_size
        freq_h, freq_w = self.freq_indices

        # 随机初始化 kernel attention
        k_att = torch.randn(b, self.kernel_num, 1, 1, device=x.device)
        k_att = torch.sigmoid(k_att / self.kernel_temp) * 2 / self.kernel_num

        # 广播并乘权重
        dft = self.dft_weight.unsqueeze(0)  # [1, kernel_num, freq, 2]
        k_att = k_att.view(b, self.kernel_num, 1, 1)  # [B, kernel_num, 1, 1]
        weighted = dft * k_att  # [B, kernel_num, freq, 2]
        w_real = weighted[..., 0].sum(dim=1)  # [B, freq]
        w_imag = weighted[..., 1].sum(dim=1)  # [B, freq]

        # 构造频域 map（注意：这里仅演示核心逻辑）
        DFT_map = torch.zeros((b, d1, d2 // 2 + 1, 2), device=x.device)
        DFT_map[:, freq_h, freq_w, 0] = w_real
        DFT_map[:, freq_h, freq_w, 1] = w_imag

        # irfft2 变换
        spatial_weight = torch.fft.irfft2(torch.view_as_complex(DFT_map), s=(d1, d2))  # [B, d1, d2]
        spatial_weight = spatial_weight.view(b, self.out_channels, self.kernel_size,
                                             self.in_channels, self.kernel_size).permute(0, 1, 3, 2, 4)

        # 应用卷积
        output = []
        for i in range(b):
            w_i = spatial_weight[i].reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            out_i = F.conv2d(x[i:i + 1], w_i, stride=self.stride, padding=self.padding)
            output.append(out_i)
        return torch.cat(output, dim=0)


if __name__ == '__main__':
    x = torch.randn(2, 32, 64, 64).cuda()
    model = FDConv(32, 64, kernel_size=3, padding=1).cuda()
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)
