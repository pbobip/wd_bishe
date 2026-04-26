import torch
import torch.nn as nn
'''
论文链接：https://arxiv.org/pdf/2303.16900
论文标题：InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)
'''

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据 branch_ratio 计算分支卷积层所使用的通道数 gc
        gc = int(in_channels * branch_ratio)

        # 对于二维卷积，常用的填充参数格式为 (height_padding, width_padding) 。
        # ai缝合大王

        # 使用深度可分离卷积处理正方形区域，采用 gc 个分组卷积
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 针对水平（W）方向的条形区域采用 1xband_kernel_size 的深度卷积，采用相同分组数 gc
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        # 针对垂直（H）方向的条形区域采用 band_kernel_size x1 的深度卷积，采用相同分组数 gc
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        # 计算各个分支的通道分割索引，剩余部分不经过卷积处理，其余三支各处理 gc 个通道
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        # ai缝合大王

    def forward(self, x):
        # 将输入张量 x 根据预先计算的通道索引分为四部分：
        # 第一部分不经过卷积，后面三个部分分别用于正方形卷积、水平卷积和垂直卷积
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        # 将各分支经过各自卷积处理后的结果与未处理部分拼接在一起，沿通道维度组合成最终输出
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

if __name__ == '__main__':
    model = InceptionDWConv2d(in_channels=64)
    input_tensor = torch.randn(1, 64, 224, 224)
    output_tensor = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    # ai缝合大王
