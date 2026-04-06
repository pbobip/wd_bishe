import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文来源：https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf
    论文标题：Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention(CVPR 2024)
"""

def get_freq_indices(method):
    """
    根据指定的方法获取 DCT 频率索引。
    """
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
    else:
        raise NotImplementedError
    return all_top_indices_x[:num_freq], all_top_indices_y[:num_freq]

class MultiFrequencyChannelAttention(nn.Module):
    """
    多频通道注意力机制（MFCA）。
    """
    def __init__(self, in_channels, dct_h=7, dct_w=7, frequency_branches=16, frequency_selection='top', reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        for freq_idx in range(frequency_branches):
            self.register_buffer(f'dct_weight_{freq_idx}', self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False))
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)
        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return result * math.sqrt(2) if freq != 0 else result

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x_pooled = x if (H == self.dct_h and W == self.dct_w) else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg /= self.num_freq
        multi_spectral_feature_max /= self.num_freq
        multi_spectral_feature_min /= self.num_freq
        multi_spectral_attention_map = torch.sigmoid(self.fc(multi_spectral_feature_avg + multi_spectral_feature_max + multi_spectral_feature_min)).view(batch_size, C, 1, 1)
        return x * multi_spectral_attention_map.expand_as(x)

if __name__ == '__main__':
    input_tensor = torch.randn(8, 64, 50, 50)
    model = MultiFrequencyChannelAttention(in_channels=64, dct_h=7, dct_w=7, frequency_branches=16, frequency_selection='top', reduction=16)
    output_tensor = model(input_tensor)
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)
