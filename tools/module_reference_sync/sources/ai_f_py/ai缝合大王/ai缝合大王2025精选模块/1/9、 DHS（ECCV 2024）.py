import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
论文链接：https://ieeexplore.ieee.org/abstract/document/10445289/
论文题目：Restoring Images in Adverse Weather Conditions via Histogram Transformer (ECCV 2024)
"""

class Dynamic_range_Histogram_SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(Dynamic_range_Histogram_SelfAttention, self).__init__()
        self.num_heads = num_heads  # 设置多头注意力中的头数
        self.factor = num_heads  # 将头数作为分块因子使用
        self.ifBox = ifBox  # 控制重排策略：True表示局部块，False表示全局结构
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 用于缩放注意力得分的温度参数
        # 利用1x1卷积将输入特征映射为5倍的通道数，以便后续分支处理
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        # 对qkv输出应用3x3深度可分离卷积来增强局部信息捕捉
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        # 投影层将注意力输出转换回原始通道数
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # ai缝合大王

    def pad(self, x, factor):
        # 获取最后一个维度的大小（通常为宽度），检查是否能被因子整除
        hw = x.shape[-1]
        # 若无法整除，计算需要在末尾补零的长度；否则不做填充
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, mode='constant', value=0)
        return x, t_pad

    def unpad(self, x, t_pad):
        # 根据填充信息恢复原始尺寸
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        # 自定义softmax：先指数化，然后除以各维度和（加1防止除零）
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        # 对倒数第二个维度计算均值与方差进行归一化处理
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)

    def reshape_attn(self, q, k, v, ifBox):
        b, _ = q.shape[:2]
        # 对q、k、v进行填充，确保其空间尺寸能被分块因子整除
        q, t_pad = self.pad(q, self.factor)
        k, _ = self.pad(k, self.factor)
        v, _ = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor

        # 根据ifBox参数选择重排模式：
        # ifBox=True：将空间维度先分成固定大小的块，有助于捕捉局部细节；
        # ifBox=False：则保持全局空间顺序以便捕捉长距离依赖。
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"

        # 重排 q, k, v 为新形状以便于注意力计算
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)

        # 对 q 和 k 进行L2归一化以稳定注意力分布
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # 计算缩放后的点积注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = attn @ v

        # 将注意力输出重排回原始的块结构
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, head=self.num_heads, b=b)
        # 移除之前添加的填充
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        # 对输入张量的前半通道沿高度方向排序，并记录排序索引
        x_sort, idx_h = x[:, :c // 2].sort(dim=-2)
        # 进一步对排序后的结果沿宽度方向排序，记录索引idx_w
        x_sort, idx_w = x_sort.sort(dim=-1)
        # 将排序后的结果赋值回输入张量的前半通道
        x[:, :c // 2] = x_sort

        # 计算qkv特征，先通过1x1卷积再进行深度卷积分离局部信息
        qkv = self.qkv_dwconv(self.qkv(x))
        # 将生成的qkv分为5个部分：q1, k1, q2, k2, v，其中每部分通道数为原始通道数
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)
        # 对v重新塑形后按最后一维排序，获取排序索引
        v, idx = v.view(b, c, -1).sort(dim=-1)
        # 使用torch.gather根据相同排序索引重排q1和k1
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 计算两组注意力输出，分别对应局部与全局信息处理
        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        # 利用torch.scatter将排序后的注意力输出恢复到原始顺序，并重塑为 (B, C, H, W)
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        # 合并局部和全局分支结果
        out = out1 * out2
        out = self.project_out(out)

        # 将输出前半部分通道替换为经过注意力处理后的值，并使用记录的排序索引还原原始空间排列
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace

        # ai缝合大王
        return out

if __name__ == "__main__":
    model = Dynamic_range_Histogram_SelfAttention(64)
    input = torch.randn(1, 64, 128, 128)
    output = model(input)
    print('Input size:', input.size())
    print('Output size:', output.size())
    # ai缝合大王
