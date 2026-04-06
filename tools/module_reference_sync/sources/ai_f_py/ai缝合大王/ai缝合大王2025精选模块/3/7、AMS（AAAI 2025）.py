import torch
import torch.nn as nn

"""
论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
论文题目：Dual Residual Attention Network for Image Denoising （2024）
"""

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 控制是否使用激活函数
        # 卷积层 + 批归一化
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x):
        x = self.conv(x)  # 执行卷积和归一化
        if self.act:
            x = self.relu(x)  # 应用激活函数（如果启用）
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

        # 前景辅助分支：通过多次上采样与卷积，生成前景掩码
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
        # 背景辅助分支：结构同前景分支
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
        # 不确定性辅助分支：生成不确定性掩码
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
        f_fg = self.cbr_fg(x)  # 前景特征提取
        f_bg = self.cbr_bg(x)  # 背景特征提取
        f_uc = self.cbr_uc(x)  # 不确定性特征提取
        mask_fg = self.branch_fg(f_fg)  # 前景掩码
        mask_bg = self.branch_bg(f_bg)  # 背景掩码
        mask_uc = self.branch_uc(f_uc)  # 不确定性掩码
        return mask_fg, mask_bg, mask_uc

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)
    SID = Semantic_Information_Decoupling(64, 64)
    output1, output2, output3 = SID(input)
    print("SID_input.shape:", input.shape)
    print("SID_output.shape:", output1.shape)
    print("SID_output.shape:", output2.shape)
    print("SID_output.shape:", output3.shape)
    # ai缝合大王
