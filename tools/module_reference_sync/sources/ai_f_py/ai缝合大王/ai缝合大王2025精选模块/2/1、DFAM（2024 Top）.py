import torch
import torch.nn as nn

'''
论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
论文标题：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
'''

class DFEM(nn.Module):
    def __init__(self, inc, outc):
        super(DFEM, self).__init__()
        # 1x1 卷积层：用于通道压缩和初步特征融合
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        # 3x3 卷积层：进一步提取局部特征
        self.Conv = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        # 上采样层：利用双线性插值将特征图尺寸扩大
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # ai缝合大王

    def forward(self, diff, accom):
        # 将两个输入特征图在通道维度上连接，融合不同信息源
        cat = torch.cat([accom, diff], dim=1)
        # 通过1x1卷积层实现通道压缩，同时叠加原始输入以形成残差连接
        cat = self.Conv_1(cat) + diff + accom
        # 使用3x3卷积进一步提取特征，并与前面输出叠加
        c = self.Conv(cat) + cat
        # 经过ReLU激活后，再次加上差异特征，突出变化信息
        c = self.relu(c) + diff
        # 对结果进行上采样，提升分辨率
        c = self.Up(c)
        return c

if __name__ == "__main__":
    # 定义输入通道数和输出通道数
    inc = 8
    outc = 8
    # 初始化 DFEM 模块，此处输入通道数为 2 * inc，输出通道数为 outc
    model = DFEM(2 * inc, outc)
    # 生成两个随机输入张量，尺寸均为 (1, inc, 32, 32)
    diff = torch.randn(1, inc, 32, 32)
    accom = torch.randn(1, inc, 32, 32)
    # 执行前向传播
    output = model(diff, accom)
    print("Output shape:", output.shape)
    # ai缝合大王
