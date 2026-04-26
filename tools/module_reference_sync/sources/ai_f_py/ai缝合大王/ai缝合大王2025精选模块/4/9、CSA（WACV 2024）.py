import torch
import torch.nn as nn
import torch.distributions as td

"""
    论文来源：https://openaccess.thecvf.com/content/WACV2024/papers/Hung_CSAM_A_2.5D_Cross-Slice_Attention_Module_for_Anisotropic_Volumetric_Medical_WACV_2024_paper.pdf
    论文标题：CSAM: A 2.5D Cross-Slice Attention Module for Anisotropic Volumetric Medical Image Segmentation（WACV 2024）
"""

# 自定义最大值函数

def custom_max(x, dim, keepdim=True):
    temp_x = x
    for i in dim:
        temp_x = torch.max(temp_x, dim=i, keepdim=True)[0]
    if not keepdim:
        temp_x = temp_x.squeeze()
    return temp_x

# 位置注意力模块
class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), padding=3)

    def forward(self, x):
        max_x = custom_max(x, dim=(0, 1), keepdim=True)
        avg_x = torch.mean(x, dim=(0, 1), keepdim=True)
        att = torch.cat((max_x, avg_x), dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att

# 语义注意力模块
class SemanticAttentionModule(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super(SemanticAttentionModule, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_rate),
            nn.ReLU(),
            nn.Linear(in_features // reduction_rate, in_features)
        )

    def forward(self, x):
        max_x = custom_max(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        att = self.linear(max_x) + self.linear(avg_x)
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x * att

# 切片注意力模块
class SliceAttentionModule(nn.Module):
    def __init__(self, in_features, rate=4, uncertainty=True, rank=5):
        super(SliceAttentionModule, self).__init__()
        self.uncertainty = uncertainty
        self.rank = rank

        self.linear = nn.Sequential(
            nn.Linear(in_features, int(in_features * rate)),
            nn.ReLU(),
            nn.Linear(int(in_features * rate), in_features)
        )

        if uncertainty:
            self.non_linear = nn.ReLU()
            self.mean = nn.Linear(in_features, in_features)
            self.log_diag = nn.Linear(in_features, in_features)
            self.factor = nn.Linear(in_features, in_features * rank)

    def forward(self, x):
        max_x = custom_max(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        att = self.linear(max_x) + self.linear(avg_x)

        if self.uncertainty:
            temp = self.non_linear(att)
            mean = self.mean(temp)
            diag = self.log_diag(temp).exp()
            factor = self.factor(temp).view(1, -1, self.rank)
            dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag)
            att = dist.sample()

        att = torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * att

# 组合注意力模块（CSAM）
class CSAM(nn.Module):
    def __init__(self, num_slices, num_channels, semantic=True, positional=True, slice=True, uncertainty=True, rank=5):
        super(CSAM, self).__init__()
        self.semantic = semantic
        self.positional = positional
        self.slice = slice

        if semantic:
            self.semantic_att = SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att = PositionalAttentionModule()
        if slice:
            self.slice_att = SliceAttentionModule(num_slices, uncertainty=uncertainty, rank=rank)

    def forward(self, x):
        if self.semantic:
            x = self.semantic_att(x)
        if self.positional:
            x = self.positional_att(x)
        if self.slice:
            x = self.slice_att(x)
        return x

if __name__ == '__main__':
    model = CSAM(num_slices=10, num_channels=64)
    input_tensor = torch.randn(10, 64, 128, 128)
    output_tensor = model(input_tensor)
    
    print('输入张量尺寸:', input_tensor.size())
    print('输出张量尺寸:', output_tensor.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')
