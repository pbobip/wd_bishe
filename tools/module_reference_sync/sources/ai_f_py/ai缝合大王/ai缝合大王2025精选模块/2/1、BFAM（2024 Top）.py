import torch
import torch.nn as nn

'''
论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
论文题目：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
'''

class simam_module(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        # 初始化 Sigmoid 激活层，用于将输出映射到 (0, 1) 区间
        self.activaton = nn.Sigmoid()
        # 设置正则化系数 lambda，用以平衡激活计算
        self.e_lambda = e_lambda
        # ai缝合大王

    def __repr__(self):
        # 返回模块名称及参数信息
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        # 返回该模块的简称
        return "simam"

    def forward(self, x):
        # 获取输入张量的尺寸 (B, C, H, W)
        b, c, h, w = x.size()
        n = h * w - 1
        # 计算每个像素与局部均值差异的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 根据公式计算调制因子 y，防止除零并加入常数偏移
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class diff_moudel(nn.Module):
    def __init__(self, in_channel):
        super(diff_moudel, self).__init__()
        # 构建 3x3 平均池化层以平滑输入特征
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        # 采用 1x1 卷积进行通道重映射
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # 使用批归一化层规范化卷积输出
        self.bn1 = nn.BatchNorm2d(in_channel)
        # Sigmoid 激活用于将数值限制在 [0, 1]
        self.sigmoid = nn.Sigmoid()
        # 实例化 SIMAM 模块以进一步增强重要特征
        self.simam = simam_module()
        
    def forward(self, x):
        # 先应用 SIMAM 调制输入
        x = self.simam(x)
        # 计算边缘信息，等于原输入与局部均值之差
        edge = x - self.avg_pool(x)
        # 利用 1x1 卷积、BN 和 Sigmoid 得到权重映射
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # ai缝合大王
        # 将权重应用于原始特征，并采用残差结构保留初始信息
        out = weight * x + x
        # 最后再利用 SIMAM 模块对输出进行细化
        out = self.simam(out)
        return out

class CBM(nn.Module):
    def __init__(self, in_channel):
        super(CBM, self).__init__()
        # 分别构建两个差异提取模块，用于处理不同输入
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        # 初始化一个额外的 SIMAM 模块用于后续特征融合
        self.simam = simam_module()

    def forward(self, x1, x2):
        # 对两个输入分别提取特征差异
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        # 计算两个特征之间的绝对差值，突出变化区域
        d = torch.abs(d1 - d2)
        # 通过 SIMAM 模块进一步强化变化信息
        d = self.simam(d)
        return d

if __name__ == "__main__":
    # 生成两个随机输入张量，形状为 (1, 32, 128, 128)
    input1 = torch.randn(1, 32, 128, 128)
    input2 = torch.randn(1, 32, 128, 128)
    # 实例化 CBM 模块，输入通道数设置为 32
    cbm = CBM(32)
    output = cbm(input1, input2)
    print('BFAM_input_size:', input1.size())
    print('BFAM_output_size:', output.size())
    # ai缝合大王
