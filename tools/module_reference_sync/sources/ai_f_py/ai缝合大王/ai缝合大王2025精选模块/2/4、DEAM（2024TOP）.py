import torch
import torch.nn.functional as F
from torch import nn

'''
论文链接：https://ieeexplore.ieee.org/abstract/document/10504297
论文标题：DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection （2024 TOP）
'''

class DEAM(nn.Module):
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(DEAM, self).__init__()
        self.chanel_in = in_dim
        # 将输入通道数降低至1/8，用于计算注意力
        self.key_channel = self.chanel_in // 8  
        self.activation = activation
        self.ds = ds  # 下采样比例因子
        # 采用平均池化进行下采样处理
        self.pool = nn.AvgPool2d(self.ds)
        # 1x1 卷积用于生成查询向量
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # 1x1 卷积用于生成键向量
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # 1x1 卷积用于生成值向量
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 可训练的缩放参数，用于调整注意力贡献
        self.gamma = nn.Parameter(torch.zeros(1))
        # 使用 softmax 在最后一维上归一化注意力得分
        self.softmax = nn.Softmax(dim=-1)
        # ai缝合大王

    def forward(self, input, diff):
        """
        输入:
            input : 主输入特征图 (B, C, H, W)
            diff  : 差异特征图 (B, C, H, W)
        输出:
            out   : 与原输入相加的注意力增强特征
        """
        # 对差异特征图进行下采样
        diff = self.pool(diff)
        m_batchsize, C, width, height = diff.size()
        # 生成查询向量，并调整维度为 (B, N, C')，其中 N=width*height
        proj_query = self.query_conv(diff).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # 生成键向量，形状为 (B, C', N)
        proj_key = self.key_conv(diff).view(m_batchsize, -1, width * height)
        # 计算点积得分矩阵
        energy = torch.bmm(proj_query, proj_key)
        # 对能量进行缩放归一化
        energy = (self.key_channel ** -0.5) * energy
        # 通过 softmax 获取注意力分布
        attention = self.softmax(energy)

        # 对输入主特征图也进行下采样
        x = self.pool(input)
        # 生成值向量，重塑为 (B, C, N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # 利用注意力矩阵对值进行加权求和
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # 将结果上采样回原始尺寸
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        # 将加权特征与原始输入相加，形成残差连接
        out = out + input
        # ai缝合大王
        return out

if __name__ == '__main__':
    x = torch.randn((8, 128, 32, 32))  # 随机生成主输入特征图
    y = torch.randn((8, 128, 32, 32))  # 随机生成差异图
    model = DEAM(128)  # 实例化 DEAM 模块，输入通道数为128
    out = model(x, y)  # 执行前向传播
    print(out.shape)  # 输出最终结果的形状
    # ai缝合大王
