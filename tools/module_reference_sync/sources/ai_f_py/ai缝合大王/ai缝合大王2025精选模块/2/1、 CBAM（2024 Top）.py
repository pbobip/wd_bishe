import torch
import torch.nn as nn

'''
论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
论文标题：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
'''

class simam_module(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        # 初始化 Sigmoid 激活层，用于将计算后的权重映射到 [0, 1] 区间
        self.activaton = nn.Sigmoid()
        # 保存正则化系数 lambda，用于平衡激活响应
        self.e_lambda = e_lambda

    def __repr__(self):
        # 返回模块的描述字符串，显示模块名称和 lambda 参数
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
        # 计算每个像素与局部均值差值的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 根据公式计算中间调制因子 y，防止除零问题
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 将原始输入与经过 Sigmoid 激活后的 y 相乘，获得加权输出
        return x * self.activaton(y)

# ai缝合大王

class diff_moudel(nn.Module):
    def __init__(self, in_channel):
        super(diff_moudel, self).__init__()
        # 构建 3x3 平均池化层以平滑局部区域特征
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        # 1x1 卷积层用于调整通道分布
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # 批归一化层用于稳定训练过程中的特征分布
        self.bn1 = nn.BatchNorm2d(in_channel)
        # Sigmoid 激活函数将卷积输出映射到 [0, 1]
        self.sigmoid = nn.Sigmoid()
        # 实例化 SIMAM 模块以增强关键区域信息
        self.simam = simam_module()
        
    def forward(self, x):
        # 首先利用 SIMAM 模块对输入进行初步调制
        x = self.simam(x)
        # 计算边缘信息，突出局部细节与变化
        edge = x - self.avg_pool(x)
        # ai缝合大王
        # 通过 1x1 卷积、批归一化和 Sigmoid 函数生成权重映射
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # 将权重与输入相乘，再加上原始输入形成残差结构，保留原有信息
        out = weight * x + x
        # 进一步利用 SIMAM 模块强化输出特征
        out = self.simam(out)
        return out

class CBM(nn.Module):
    def __init__(self, in_channel):
        super(CBM, self).__init__()
        # 初始化两个差异提取模块，分别对不同输入进行处理
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        # 再次利用 SIMAM 模块对融合后的差异信息进行调节
        self.simam = simam_module()

    def forward(self, x1, x2):
        # 分别提取两个输入的差异特征
        d1 = self.diff_1(x1)
        d2 = self.diff_moudel(x2) if False else self.diff_2(x2)  # 这里保持 diff_2(x2) 的调用
        # 计算两组特征的绝对差值
        d = torch.abs(d1 - d2)
        # 通过 SIMAM 模块细化差异信息
        d = self.simam(d)
        return d

if __name__ == "__main__":
    # 生成两个随机输入，尺寸均为 (1, 32, 128, 128)
    input1 = torch.randn(1, 32, 128, 128)
    input2 = torch.randn(1, 32, 128, 128)
    # 实例化 CBM 模块，设定通道数为 32
    cbm = CBM(32)
    output = cbm(input1, input2)
    print('CBM_input_size:', input1.size())
    print('CBM_output_size:', output.size())
    # ai缝合大王
