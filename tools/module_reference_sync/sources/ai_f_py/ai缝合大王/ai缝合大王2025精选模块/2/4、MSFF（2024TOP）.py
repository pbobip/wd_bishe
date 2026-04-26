import torch
import torch.nn as nn

'''
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
论文题目：Dual Residual Attention Network for Image Denoising （2024）
'''

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        # 第一分支：利用1x1卷积实现简单的特征通道保留和变换
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 第二分支：采用1x1卷积降维，3x3卷积提取局部特征，随后1x1卷积恢复通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 第三分支：利用5x5卷积核提取较大感受野的特征
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 第四分支：使用7x7卷积核获取更宽广的上下文信息
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # ai缝合大王
        # 混合卷积层：将来自四个分支的输出在通道维度上连接后进行特征整合
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 分别通过四个不同尺寸卷积分支处理输入
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        # 在通道维度上串联所有分支的输出
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        # 使用混合卷积层融合多尺度特征
        out = self.convmix(x_f)
        return out

if __name__ == '__main__':
    # 构造一个批量为32，通道256，尺寸32x32的随机输入
    x = torch.randn((32, 256, 32, 32))
    # 实例化 MSFF 模块，设定输入通道256，降维中间通道64
    model = MSFF(256, 64)
    # 执行前向传播
    out = model(x)
    # ai缝合大王
    print(out.shape)
