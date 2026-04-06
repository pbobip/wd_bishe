import math
import torch
import torch.nn as nn

"""
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
论文题目：Agent Attention: On the Integration of Softmax and Linear Attention (ECCV 2024)
"""

class ESSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 创建一个全连接层，将输入特征映射到三倍维度，用于生成查询、键和值
        self.lnqkv = nn.Linear(dim, dim * 3)
        # 创建输出投影层，将融合后的特征映射回原始维度
        self.ln = nn.Linear(dim, dim)
        # ai缝合大王

    def forward(self, x):
        # 获取输入的批次、通道、高度和宽度
        b, c, h, w = x.shape
        # 将输入张量展平成 (B, C, H*W) 后转置为 (B, H*W, C)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        b, N, C = x.shape
        # 利用全连接层生成查询、键和值，结果形状为 (B, N, 3 * C)
        qkv = self.lnqkv(x)
        # 将qkv拆分成三部分，每部分的形状为 (B, N, C)
        q, k, v = torch.split(qkv, C, dim=2)

        # 计算查询向量的均值并执行中心化处理
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        # 对键向量也做同样的中心化处理
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a

        # 计算查询和键的平方，用于后续归一化
        q2 = q.pow(2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = k.pow(2)
        k2s = torch.sum(k2, dim=2, keepdim=True)

        # 第一部分输出直接使用值向量
        t1 = v
        # 对键的平方进行归一化，保证数值稳定性
        k2 = torch.nn.functional.normalize(q2 / (q2s + 1e-7), dim=-1)  # Note: 原代码归一化顺序与说明略有不同
        # 对查询的平方进行归一化
        q2 = torch.nn.functional.normalize(k2 / (k2s + 1e-7), dim=-2)
        # 计算第二部分输出，通过归一化后的查询和键计算加权值，并除以sqrt(N)进行缩放
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        # 合并两部分输出
        attn = t1 + t2
        # 利用输出投影层映射回原始维度
        attn = self.ln(attn)
        # 重构为 (B, h, w, C)，再转置回 (B, C, h, w)
        x = attn.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x

if __name__ == '__main__':
    # 创建一个随机输入张量，形状为 (1, 32, 77, 77)
    input = torch.randn(1, 32, 77, 77)
    model = ESSA(32)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
    # ai缝合大王
