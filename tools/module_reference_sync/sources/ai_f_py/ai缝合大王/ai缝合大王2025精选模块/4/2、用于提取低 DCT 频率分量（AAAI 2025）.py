import torch
import torch.nn as nn
import torch.fft
import math

"""
    论文来源：https://arxiv.org/abs/2412.13753
    论文标题：Mesoscopic Insights: Orchestrating Multi-scale & Hybrid Architecture for Image Manipulation Localization(AAAI 2025)
"""

# 定义 LowDctFrequencyExtractor 类，用于提取低频 DCT 分量
class LowDctFrequencyExtractor(nn.Module):
    def __init__(self, alpha=0.95):
        super(LowDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha 必须在 (0,1) 之间")
        self.alpha = alpha
        self.dct_matrix_h = None  # 水平方向的 DCT 变换矩阵
        self.dct_matrix_w = None  # 垂直方向的 DCT 变换矩阵

    # 生成 DCT 变换矩阵
    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)  # 归一化处理
        return dct_matrix

    # 计算二维 DCT 变换
    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    # 计算二维逆 DCT 变换
    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    # 低通滤波，去除高频分量，保留低频信息
    def low_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[-alpha_h:, -alpha_w:] = 0  # 屏蔽高频部分
        return x * mask

    # 前向传播，提取低频信息
    def forward(self, x):
        xq = self.dct_2d(x)  # 计算 DCT 变换
        xq_low = self.low_pass_filter(xq, self.alpha)  # 低通滤波
        xh = self.idct_2d(xq_low)  # 计算逆 DCT 变换

        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)  # 归一化到 [0, 1]
        return xh

if __name__ == '__main__':
    input_tensor = torch.rand(1, 64, 32, 32)  # 创建随机输入张量
    extractor = LowDctFrequencyExtractor()  # 创建 LowDctFrequencyExtractor 实例
    output = extractor(input_tensor)  # 处理输入张量
    print(input_tensor.size())  # 打印输入张量的形状
    print(output.size())  # 打印输出张量的形状