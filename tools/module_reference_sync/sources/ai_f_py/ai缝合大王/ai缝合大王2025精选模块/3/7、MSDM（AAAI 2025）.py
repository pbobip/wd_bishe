import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文地址：https://arxiv.org/pdf/2406.03751
论文题目：Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting（AAAI 2025）
"""

class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.1, layernorm=True):
        super(DDI, self).__init__()
        # input_shape: (seq_len, feature_num)
        self.input_shape = input_shape
        # 如果alpha > 0, 则构造前馈模块，其隐藏层维度为输入特征数向上取整为2的幂
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.n_history = 1
        self.alpha = alpha
        self.patch = patch  # 每个块的长度

        self.layernorm = layernorm
        if self.layernorm:
            # 对展平后的时间序列进行批归一化
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        # 对每个历史块进行归一化
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])
        # 聚合层将历史块整合到当前块（线性映射）
        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        output = torch.zeros_like(x)
        # 将初始的n_history*patch部分直接复制到输出中
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()

        # 从索引n_history*patch开始，每隔patch处理一块
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # 取出上一历史块作为输入
            curr_block = output[:, :, i - self.n_history * self.patch : i]
            curr_block = self.norm1(torch.flatten(curr_block, 1, -1)).reshape(curr_block.shape)
            # 聚合历史块信息，生成当前块预测
            agg_block = F.gelu(self.agg(curr_block))
            agg_block = self.dropout_t(agg_block)
            # 残差：当前聚合块加上原始对应块
            tmp = agg_block + x[:, :, i: i + self.patch]
            res = tmp

            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp

        return output

if __name__ == "__main__":
    input_shape = (48, 10)  # 序列长度=48, 特征数量=10
    model = DDI(input_shape=input_shape, dropout=0.2, patch=12, alpha=0.1, layernorm=True)
    batch_size = 16
    # 输入形状: [batch_size, feature_num, seq_len]
    x = torch.randn(batch_size, input_shape[1], input_shape[0])
    print("Input shape:", x.shape)
    output = model(x)
    print("Output shape:", output.shape)
