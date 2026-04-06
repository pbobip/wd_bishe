import torch
import torch.nn as nn

"""
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
论文题目：Dual Residual Attention Network for Image Denoising (2024)
"""

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 标志是否应用激活函数
        # 构造卷积层和批归一化层的组合
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 使用ReLU增加非线性

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class Semantic_Information_Decoupling(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(Semantic_Information_Decoupling, self).__init__()
        # 前景特征提取分支
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        # 背景特征提取分支
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        # 不确定性特征提取分支
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

        # 前景辅助分支，通过上采样和1x1卷积生成前景掩码
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        # 背景辅助分支，结构同上
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        # 不确定性辅助分支
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取前景、背景及不确定性特征
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        # 利用辅助分支生成对应掩码
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc

if __name__ == '__main__':
    # 创建随机输入张量，形状为 (1, 64, 32, 32)
    input = torch.rand(1, 64, 32, 32)
    SID = Semantic_Information_Decoupling(64, 64)
    output1, output2, output3 = SID(input)
    print("SID_input.shape:", input.shape)
    print("SID_output.shape:", output1.shape)
    print("SID_output.shape:", output2.shape)
    print("SID_output.shape:", output3.shape)
    # ai缝合大王
