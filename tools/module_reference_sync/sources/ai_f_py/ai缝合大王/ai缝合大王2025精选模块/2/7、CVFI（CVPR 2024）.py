import torch
import torch.nn as nn

"""
论文地址：https://ieeexplore.ieee.org/abstract/document/10504297
论文题目：Multi-Level Feature Fusion Network for Lightweight Stereo Image Super-Resolution (CVPR 2024)
"""

class CVIM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5  # 缩放因子，用于归一化特征内积，避免数值过大

        # 左侧第一组卷积层：由1x1卷积和3x3深度卷积构成，提取局部特征
        self.l_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        # 右侧第一组卷积层：结构同左侧
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        # 左侧第二组卷积层：进一步处理左侧特征
        self.l_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        # 右侧第二组卷积层：进一步处理右侧特征
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        # 左侧第三个卷积层：使用1x1卷积实现通道融合
        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        # 右侧第三个卷积层：同样采用1x1卷积
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        # 提取左侧特征，并转换为 (B, H, W, c) 格式
        Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)
        # 提取右侧特征，并转换为 (B, H, c, W) 格式，为矩阵乘法做准备
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)
        # 计算点积注意力矩阵，并进行缩放
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        # 提取左侧和右侧第二组特征，分别转换为 (B, H, W, c)
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)
        # 利用 softmax 和注意力矩阵计算从右到左的特征映射
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)
        # 利用转置后的注意力计算从左到右的特征映射
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)
        # ai缝合大王

        # 通过1x1卷积将转换后的注意力特征恢复到原始格式
        F_r2l = self.l_proj3(F_r2l.permute(0, 3, 1, 2))
        F_l2r = self.r_proj3(F_l2r.permute(0, 3, 1, 2))
        # 最终输出为左侧输入、右侧输入以及双向映射特征的叠加
        return x_l + F_r2l + x_r + F_l2r

if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)  # 随机生成左侧输入张量
    input2 = torch.randn(1, 32, 64, 64)  # 随机生成右侧输入张量
    CVIM_module = CVIM(32)  # 初始化 CVIM 模块，设定通道数为32
    output = CVIM_module(input1, input2)  # 执行前向传播
    print("Input size:", input1.shape, input2.shape)
    print("Output size:", output.shape)
    # ai缝合大王
