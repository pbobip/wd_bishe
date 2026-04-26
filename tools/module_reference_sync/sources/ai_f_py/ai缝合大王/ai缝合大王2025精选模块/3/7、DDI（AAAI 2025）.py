import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文链接：https://arxiv.org/pdf/2406.03751
论文题目：Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting（AAAI 2025）
"""

class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        super(DDI, self).__init__()
        # 保存输入尺寸信息：第一个元素为序列长度，第二个为特征数
        self.input_shape = input_shape
        if alpha > 0.0:
            # 计算前馈网络的隐藏维度，向上取整为2的幂次
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        # 历史信息使用一个时间步
        self.n_history = 1
        self.alpha = alpha  # 控制额外前馈网络的残差比例
        self.patch = patch  # 每个块的长度

        self.layernorm = layernorm  # 是否执行层归一化
        if self.layernorm:
            # 使用1d批归一化，对展平后的序列进行归一化处理
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        # 聚合层将历史块信息压缩到当前块维度
        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)
        # ai缝合大王

    def forward(self, x):
        # 输入x的形状为 [batch_size, feature_num, seq_len]
        if self.layernorm:
            # 对输入先展平后进行归一化，再还原形状
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        # 初始化输出为与x相同的张量
        output = torch.zeros_like(x)
        # 复制初始历史块到输出中
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()

        # 遍历序列，以patch为步长，逐块处理时间序列
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # 选取上一段的历史块作为输入
            curr_block = output[:, :, i - self.n_history * self.patch: i]
            # 将块展平并归一化后重塑为原始形状
            curr_block = self.norm1(torch.flatten(curr_block, 1, -1)).reshape(curr_block.shape)
            # 利用聚合层将历史信息整合到一个块中，并激活
            agg_block = F.gelu(self.agg(curr_block))
            agg_block = self.dropout_t(agg_block)

            # 计算当前块的残差
            tmp = agg_block + x[:, :, i: i + self.patch]
            res = tmp  # 保存中间结果

            # 如果alpha大于零，则通过额外的前馈层进行调整
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)  # 交换维度以适配fc_block输入
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)  # 还原原始维度顺序
            # 更新当前块的输出，加上前馈模块调整结果（按比例alpha）
            output[:, :, i: i + self.patch] = res + self.alpha * tmp
            # ai缝合大王

        return output

if __name__ == "__main__":
    # 示例输入：序列长度为48，特征数为10
    input_shape = (48, 10)
    dropout = 0.2
    patch = 12
    alpha = 0.1
    layernorm = True

    model = DDI(input_shape=input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
    batch_size = 16
    # 输入张量形状：[batch_size, feature_num, seq_len]
    x = torch.randn(batch_size, input_shape[1], input_shape[0])
    print("Input shape:", x.shape)
    output = model(x)
    print("Output shape:", output.shape)
    # ai缝合大王
