import torch
import torch.nn as nn

'''
论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
论文标题：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
'''

class simam_module(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        # 初始化 Sigmoid 激活，用以将输出映射到 (0, 1)
        self.activaton = nn.Sigmoid()
        # 设置正则化常数 lambda，用于平衡激活函数输入
        self.e_lambda = e_lambda
        # ai缝合大王

    def __repr__(self):
        # 返回模块名称及参数信息
        return f"{self.__class__.__name__}(lambda={self.e_lambda:.6f})"

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        # 获取输入张量尺寸: Batch, Channel, Height, Width
        b, c, h, w = x.size()
        n = h * w - 1
        # 计算每个像素与全局均值之差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 生成调制因子 y，确保数值稳定性
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class BFAM(nn.Module):
    def __init__(self, inp, out):
        super(BFAM, self).__init__()

        # 初始化两个 SIMAM 模块，用于预处理和后续特征调制
        self.pre_siam = simam_module()
        self.lat_siam = simam_module()
        # ai缝合大王

        # 保留输入特征的通道数作为输出的基准
        out_1 = inp
        # 更新输入通道数为两者之和，用于后续融合
        inp = inp + out

        # 定义一组膨胀卷积层，分别使用不同的空洞率捕捉多尺度信息
        self.conv_1 = nn.Conv2d(inp, out_1, kernel_size=3, padding=1, dilation=1, groups=out_1)
        self.conv_2 = nn.Conv2d(inp, out_1, kernel_size=3, padding=2, dilation=2, groups=out_1)
        self.conv_3 = nn.Conv2d(inp, out_1, kernel_size=3, padding=3, dilation=3, groups=out_1)
        self.conv_4 = nn.Conv2d(inp, out_1, kernel_size=3, padding=4, dilation=4, groups=out_1)

        # 融合层：利用1x1卷积、BN和ReLU将多个尺度的卷积输出整合
        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )

        # 再次使用 SIMAM 模块调制融合结果
        self.fuse_siam = simam_module()

        # 输出层，将融合后的特征映射为所需通道数
        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp1, inp2):
        # 初始化 last_feature（预留用于其他情形，但此处始终为 None）
        last_feature = None

        # 将两个输入特征图在通道上拼接
        x = torch.cat([inp1, inp2], dim=1)

        # 分别通过四个膨胀卷积层提取不同尺度的特征
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        # 将四组特征在通道维度上合并
        cat = torch.cat([c1, c2, c3, c4], dim=1)
        # 利用融合层整合多尺度信息
        fuse = self.fuse(cat)

        # 对两个输入分别进行 SIMAM 调制
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)
        # 将 SIMAM 调制结果与融合结果逐元素相乘
        inp1_mul = torch.mul(inp1_siam, fuse)
        inp2_mul = torch.mul(inp2_siam, fuse)

        # 对融合特征再次应用 SIMAM 模块
        fuse = self.fuse_siam(fuse)

        # 根据 last_feature 状态构造残差输出
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse + inp2_mul + inp1_mul + last_feature + inp1 + inp2)

        # 最终利用 SIMAM 模块进一步调节输出特征
        out = self.fuse_siam(out)
        return out

if __name__ == "__main__":
    # 生成两个随机张量作为输入，尺寸均为 (1, 30, 128, 128)
    input1 = torch.randn(1, 30, 128, 128)
    input2 = torch.randn(1, 30, 128, 128)
    # 初始化 BFAM 模块，设置输入和输出通道数均为 30
    bfam = BFAM(30, 30)
    output = bfam(input1, input2)
    print('BFAM_input_size:', input1.size())
    print('BFAM_output_size:', output.size())
    # ai缝合大王
