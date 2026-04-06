import torch
import torch.nn as nn

"""
    论文来源：https://arxiv.org/abs/2203.10489
    论文标题：TVConv: Efﬁcient Translation Variant Convolution for Layout-aware Visual Processing(CVPR 2022)
"""

class _ConvBlock(nn.Sequential):
    """
    卷积块：包含卷积层、层归一化（LayerNorm）和 ReLU 激活。
    """
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2  # 计算填充大小，确保输出大小与输入相同
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.LayerNorm([out_planes, h, w]),
            nn.ReLU(inplace=True)
        )

class TVConv(nn.Module):
    """
    位置变体卷积（TVConv）模块。
    """
    def __init__(self, channels, TVConv_k=3, stride=1, TVConv_posi_chans=4,
                 TVConv_inter_chans=64, TVConv_inter_layers=3, TVConv_Bias=False,
                 h=3, w=3, **kwargs):
        super(TVConv, self).__init__()
        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k**2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))
        self.bias_layers = None
        out_chans = self.TVConv_k_square * self.channels  # 计算输出通道数

        # 位置映射参数
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)

        # 创建权重层和可选的偏置层
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)
        
        # Unfold 层用于提取局部区域
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k - 1) // 2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        """
        生成卷积层序列。
        """
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for _ in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(inter_chans, out_chans, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播：计算卷积权重并应用位置变体卷积。
        """
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w)
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w)
        out = (weight * out).sum(dim=2)  # 计算加权和，相当于位置特定的卷积核
        if self.bias_layers is not None:
            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out

if __name__ == "__main__":
    input_tensor = torch.rand(2, 64, 32, 32)  # 生成输入张量 (N, C, H, W)
    model = TVConv(64, h=32, w=32)  # 实例化 TVConv 模型
    output_tensor = model(input_tensor)  # 运行 TVConv
    
    print('输入张量尺寸:', input_tensor.size())
    print('输出张量尺寸:', output_tensor.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params / 1e6:.2f}M')
