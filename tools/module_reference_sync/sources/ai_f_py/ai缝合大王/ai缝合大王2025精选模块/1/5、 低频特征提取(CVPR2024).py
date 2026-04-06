import numbers
import pywt
import torch.nn.functional as F
from torch.autograd import Function

"""
论文地址：https://arxiv.org/abs/2404.13537
文章标题：Bracketing Image Restoration and Enhancement with High-Low Frequency Decomposition (CVPR 2024)
"""

import torch
import torch.nn as nn
from einops import rearrange  # 用于重组张量维度

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # 若传入的 normalized_shape 为单个整数，则转换为单元素元组形式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        # 确保归一化尺寸只有一个维度
        assert len(normalized_shape) == 1
        # 初始化权重参数，全部设为 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算最后一维的方差
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 返回归一化并缩放后的张量
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        # 将整数形式的 normalized_shape 转换为元组
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        # 确保 normalized_shape 为单维度
        assert len(normalized_shape) == 1
        # 初始化归一化的权重与偏置
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算均值与方差（最后一维）
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 返回归一化、缩放并平移后的结果
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    # 将4D张量转换为3D格式：批次, 像素数, 通道数
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    # 将3D张量重排为4D格式，恢复指定的高度和宽度
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # 根据类型选择无偏或有偏的 LayerNorm 实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # 对输入先重排为3D，归一化后再恢复成4D
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        # 第一个1x1卷积将输入扩展到 2 倍隐藏特征数
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度卷积层，使用3x3卷积提取局部信息，输出通道数保持一致
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        # 第二个1x1卷积将通道数还原回原始维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # 将深度卷积的输出沿通道拆分成两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 使用 GELU 激活 x1，并与 x2 逐元素相乘
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 确保输入连续存储
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        # 对输入进行卷积，分别提取低低、低高、高低、高高子带
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个子带在通道维度拼接
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            # 分离滤波器各部分并计算对应子带的梯度
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 利用 pywt 获取逆小波变换滤波器
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        # 生成低低、低高、高低、高高滤波器矩阵
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        # 扩展维度以符合卷积核要求
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 使用 pywt 生成小波分解滤波器，并反转序列顺序
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # 构造各个子带滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        # 注册滤波器为缓冲区变量
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        # 定义两个3x3卷积层，保持通道数不变
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        # 添加跳跃连接，将输入与卷积输出相加
        out2 += x
        return out2

class Fusion(nn.Module):
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        # 初始化二维离散小波变换模块
        self.dwt = DWT_2D(wave)
        # 利用1x1卷积将高频部分通道数从 in_channels*3 降为 in_channels
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 使用残差网络提取高频特征
        self.high = ResNet(in_channels)
        # 还原高频特征至原始通道数的三倍
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        # 将低频信息拼接后通过1x1卷积压缩至 in_channels
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 使用残差网络提取低频特征
        self.low = ResNet(in_channels)
        # 初始化二维逆离散小波变换模块
        self.idwt = IDWT_2D(wave)
        # ai缝合大王

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        # 对 x1 进行小波分解，获得四个子带，分别对应低频与高频信息
        x_dwt = self.dwt(x1)  # 输出形状：[B, in_channels*4, H/2, W/2]
        # 将分解结果按通道拆分为 ll、lh、hl、hh 四个部分
        ll, lh, hl, hh = x_dwt.split(c, 1)
        # 将高频部分（lh, hl, hh）拼接在一起
        high = torch.cat([lh, hl, hh], 1)  # 输出形状：[B, in_channels*3, H/2, W/2]
        high1 = self.convh1(high)
        high2 = self.high(high1)
        highf = self.convh2(high2)
        # 获取低频部分 ll 与辅助输入 x2 的尺寸
        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape
        # 若高度不匹配，则对 x2 进行上方填充以对齐
        if h1 != h2:
            x2 = F.pad(x2, (0, 0, 1, 0), "constant", 0)
        low = torch.cat([ll, x2], 1)  # 拼接后形状：[B, in_channels*2, H/2, W/2]
        low = self.convl(low)
        lowf = self.low(low)
        # 合并高频与低频特征
        out = torch.cat((lowf, highf), 1)  # 输出形状：[B, in_channels*4, H/2, W/2]
        out_idwt = self.idwt(out)  # 逆小波变换后恢复至原始尺寸：[B, in_channels, H, W]
        return out_idwt

class Channe_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Channe_Attention, self).__init__()
        self.num_heads = num_heads
        # 初始化多头注意力的缩放参数，用于调节点积结果
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 通过1x1卷积生成查询、键和值的拼接张量
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 深度卷积对 qkv 特征进行局部处理
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 输出投影层将注意力输出映射回原始维度
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # ai缝合大王

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # 将 qkv 分离为查询 Q、键 K 与值 V
        q, k, v = qkv.chunk(3, dim=1)
        # 重排张量以适应多头注意力格式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 对查询与键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # 第一次归一化后接注意力模块
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Channe_Attention(dim, num_heads, bias)
        # 第二次归一化后接前馈网络
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 残差连接：归一化 -> 注意力 -> 加原始输入
        x = x + self.attn(self.norm1(x))
        # 残差连接：归一化 -> 前馈网络 -> 加上一步结果
        x = x + self.ffn(self.norm2(x))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, wave):
        super(UNet, self).__init__()
        # 构造三个 Transformer 模块作为编码器部分
        self.trans1 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        self.trans2 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        self.trans3 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        # 使用平均池化进行下采样
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        # 利用 Fusion 模块实现上采样和特征融合
        self.upsample1 = Fusion(in_channels, wave)
        self.upsample2 = Fusion(in_channels, wave)

    def forward(self, x):
        x1 = x  # 保存原始输入，尺寸为 torch.Size([B, 32, 64, 64])
        x1_r = self.trans1(x)  # 第一 Transformer 模块输出
        x2 = self.avgpool1(x1)  # 下采样到 torch.Size([B, 32, 32, 32])
        x2_r = self.trans2(x2)  # 第二 Transformer 模块输出
        x3 = self.avgpool2(x2)  # 进一步下采样至 torch.Size([B, 32, 16, 16])
        x3_r = self.trans3(x3)  # 第三 Transformer 模块输出
        x4 = self.upsample1(x2_r, x3_r)  # 上采样并融合特征，输出尺寸 torch.Size([B, 32, 32, 32])
        out = self.upsample2(x1_r, x4)     # 融合第一层特征与上采样结果，恢复至 torch.Size([B, 32, 64, 64])
        b1, c1, h1, w1 = out.shape
        b2, c2, h2, w2 = x.shape
        # 若输出高度与输入不一致，则进行必要的填充
        if h1 != h2:
            out = F.pad(out, (0, 0, 1, 0), "constant", 0)
        # 跳跃连接：将上采样结果与原始输入相加
        X = out + x
        return X

if __name__ == '__main__':
    # 构造 UNet 模型实例，设置输入通道数为 32，采用 'haar' 小波变换
    model = UNet(32, wave='haar')
    # 生成形状为 (1, 32, 64, 64) 的随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
