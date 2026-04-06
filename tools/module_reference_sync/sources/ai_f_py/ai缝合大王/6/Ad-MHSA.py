
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter


import math

from torch.nn.functional import linear
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

# 论文：FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection

class Adaptive_MHSA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, ln=False, batch_first=True):
        super(Adaptive_MHSA, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        self.in_proj_weight = Parameter(torch.empty((3 * dim, dim)))
        self.in_proj_bias = Parameter(torch.empty(3 * dim))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = NonDynamicallyQuantizableLinear(dim, dim, bias=True)
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, Q, K, V, drop=0.5):
        B, T, D = Q.shape
        L_out = int(T * drop)
        # 通过线性变换获得 Q, K, V
        Q, K, V = linear(Q, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        dim_split = self.dim_LIN // self.num_heads
        # 分头并调整维度
        Q = Q.view(B, T, self.num_heads, dim_split).transpose(1, 2).contiguous().view(B * self.num_heads, T, dim_split)
        K = K.view(B, T, self.num_heads, dim_split).transpose(1, 2).contiguous().view(B * self.num_heads, T, dim_split)
        V = V.view(B, T, self.num_heads, dim_split).transpose(1, 2).contiguous().view(B * self.num_heads, T, dim_split)
        Q = Q / math.sqrt(dim_split)
        # 计算注意力权重
        A = torch.softmax(torch.bmm(Q, K.transpose(1, 2)), dim=2)
        A = self.dropout(A)

        if drop != 1:
            # 计算每个头中各位置的最大注意力得分，再将各头求和
            weight = A.view(B, self.num_heads, T, T).max(1)[0]
            weight = weight.sum(1)  # shape: (B, T)
            # 根据注意力得分选择部分token（取 topk，下采样比例为 drop）
            sampled_inds = torch.topk(weight, int(weight.shape[-1] * drop), dim=-1)[1]
            # 将 A 从 (B*num_heads, T, T) 调整为 (B, num_heads, T, T)，再根据 sampled_inds 采样
            A = torch.gather(
                A.unflatten(0, (-1, self.num_heads)), 
                2, 
                sampled_inds[:, None, :, None].repeat(1, self.num_heads, 1, T)
            ).flatten(0, 1)
        else:
            weight = None
            sampled_inds = None
        # 注意力加权聚合
        O = torch.bmm(A, V)
        # 恢复多头结构
        O = O.view(B, self.num_heads, L_out, dim_split).transpose(1, 2).contiguous().view(B, L_out, self.dim_LIN)
        O = linear(O, self.out_proj.weight, self.out_proj.bias)
        return O, weight, sampled_inds


if __name__ == '__main__':
    # B=2, N=16, C=64
    B, N, C = 2, 16, 64
    # 这里示例把 Q/K/V 全部用同一个输入，若需要也可分别命名为 input_Q/input_K/input_V
    input = torch.randn(B, N, C)

    model = Adaptive_MHSA(dim=C, num_heads=8, dropout=0.1)
    output, weight, sampled_inds = model(input, input, input, drop=0.5)

    print("input size:", input.size())  # (B, N, C)
    print("output size:", output.size())  # (B, int(N*drop), C)

