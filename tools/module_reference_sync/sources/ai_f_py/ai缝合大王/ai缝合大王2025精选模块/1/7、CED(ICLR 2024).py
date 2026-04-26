import torch  # 引入 PyTorch 库
import torch.nn as nn  # 引入神经网络模块
from timm.models.layers import trunc_normal_, DropPath, to_2tuple  # 从 timm 库中导入常用工具
act_layer = nn.ReLU  # 定义激活函数为 ReLU
# ai缝合大王
ls_init_value = 1e-6  # 设置 gamma 参数的初始值

'''
论文链接：https://arxiv.org/abs/2302.06052
论文题目：CEDNET: A CASCADE ENCODER-DECODER NETWORK FOR DENSE PREDICTION (ICLR 2024)
'''

class CED(nn.Module):
    def __init__(self, dim, drop_path=0., **kwargs):
        super().__init__()  # 初始化父类

        # 通过深度可分离卷积进行空间特征交互
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 使用批归一化稳定训练过程
        self.norm = nn.BatchNorm2d(dim)
        # 利用逐点全连接层扩展特征维度，保留空间信息
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()  # 应用激活函数
        # 逐点全连接层将通道数还原到原始维度
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # 定义可训练缩放参数 gamma，用于调节全连接层输出
        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        # ai缝合大王
        # DropPath 机制用于随机丢弃部分路径以防过拟合；当 drop_path=0 时，直接传递输入
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # ai缝合大王

    def forward(self, x):
        input = x  # 保存原始输入以供残差连接

        x = self.dwconv(x)  # 执行 7x7 深度卷积
        x = self.norm(x)    # 批归一化

        # 调整维度以适配全连接层，(N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)  # 逐点扩展特征维度
        x = self.act(x)      # 激活
        x = self.pwconv2(x)  # 逐点还原特征维度
        if self.gamma is not None:
            x = self.gamma * x  # 利用 gamma 缩放输出
        # 恢复为 (N, C, H, W) 格式
        x = x.permute(0, 3, 1, 2)

        # 结合残差连接和 DropPath 随机丢弃路径机制
        x = input + self.drop_path(x)
        return x

if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)  # 构造随机输入，尺寸为 (1, 64, 32, 32)
    model = CED(64)  # 实例化 CED 模块，设定特征通道数为 64
    output = model(input)  # 执行前向传播
    print('input_size:', input.size())  # 输出输入张量尺寸
    print('output_size:', output.size())  # 输出处理后张量尺寸


