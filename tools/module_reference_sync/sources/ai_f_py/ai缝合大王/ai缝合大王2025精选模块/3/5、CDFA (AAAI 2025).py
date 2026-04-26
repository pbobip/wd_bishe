import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文链接：https://arxiv.org/pdf/2407.05497
论文题目：Multi-Level Feature Fusion Network for Lightweight Stereo Image Super-Resolution (CVPR 2024)
"""

class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 本模块输出特征的维度
        dim = out_c
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        # 缩放因子，用于稳定点积注意力分数
        self.scale = self.head_dim ** -0.5

        # 线性变换层，用于生成值特征
        self.v = nn.Linear(dim, dim)
        # 两个线性层分别生成前景和背景的注意力权重
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层，将融合后的特征映射回原始维度
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Unfold模块将图像划分为局部块以进行局部注意力计算
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        # 平均池化下采样，确保局部块维度一致
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        # 预处理模块：两个CBR层用于初步特征提取
        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        # 后处理模块：两个CBR层用于细化融合后的特征
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        # 对输入进行预处理
        x = self.input_cbr(x)
        # 将输入转换为 (B, H, W, C) 格式，便于后续线性运算
        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        # 生成值特征，并转换为 (B, C, H, W) 格式
        v = self.v(x).permute(0, 3, 1, 2)

        # 利用Unfold将值特征拆分成局部块，重塑为 (B, num_heads, head_dim, K*K, L) 后转置
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, -1).permute(0, 1, 4, 3, 2)
        # 计算前景注意力权重
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        # 应用前景注意力权重以加权局部块
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        # 对加权后的前景特征再次利用Unfold拆分处理，得到背景分支输入
        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, -1).permute(0, 1, 4, 3, 2)
        # 计算背景注意力权重
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        # 应用背景注意力权重
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)
        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        # 后处理模块进一步整合并细化特征
        out = self.output_cbr(x_weighted_bg)
        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        # 根据特征类型选择相应的注意力线性层
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        # 计算下采样后的尺寸
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        # 对特征图进行池化后转换回 (B, H, W, C) 格式
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # 通过线性层映射后重塑为 (B, num_heads, L, K*K, K*K)
        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        # 利用注意力矩阵加权局部窗口特征，并重排为 (B, dim*K*K, L)
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, self.dim * self.kernel_size * self.kernel_size, -1)
        # 将重排后的块利用fold还原为 (B, dim, H, W)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # 通过线性投影和Dropout处理
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

# 这里提供一个简单的CBR实现，包含卷积、批归一化和ReLU激活
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    # 实例化ContrastDrivenFeatureAggregation模块
    cdfa = ContrastDrivenFeatureAggregation(in_c=64, out_c=64)
    # 创建随机输入张量及前景、背景特征图
    x = torch.randn(1, 64, 32, 32)
    fg = torch.randn(1, 64, 32, 32)  # 前景特征图
    bg = torch.randn(1, 64, 32, 32)  # 背景特征图
    output = cdfa(x, fg, bg)
    print("input shape:", x.shape)
    print("fg shape:", fg.shape)
    print("bg shape:", bg.shape)
    print("Output shape:", output.shape)
    # ai缝合大王
