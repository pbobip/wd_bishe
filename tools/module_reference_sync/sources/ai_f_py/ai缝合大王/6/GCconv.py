import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
# 论文：IMPROVED VIDEO VAE FOR LATENT VIDEO DIFFUSION MODEL

class Group_Causal_Conv3d(nn.Conv3d): 
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=(1,1,1), t_c=1):
        super().__init__(in_dim, out_dim, kernel, stride=stride, padding=padding)
        # 根据原始padding构造各方向需要的填充
        self._padding = (self.padding[2], self.padding[2],
                         self.padding[1], self.padding[1],
                         self.padding[0], self.padding[0])
        self._padding1 = (self.padding[2], self.padding[2],
                          self.padding[1], self.padding[1],
                          0, 0)
        self._padding2 = (0, 0, 0, 0, 0, self.padding[0])
        # 将padding置零，以便后续直接使用 F.pad 进行填充
        self.padding = (0, 0, 0) 
        self.t_c = t_c

    def forward(self, x, cache_x=None):   
        if self._padding[4] == 0:  
            return super().forward(F.pad(x, self._padding))
        else:
            assert self._padding[4] == 1
            b, c, t, h, w = x.shape 
            
            if t == 1: 
                x = F.pad(x, self._padding1)
                # 当时间步为1时复制一份沿时间轴拼接，使得后续卷积正常执行
                x = torch.cat([x, x], dim=2)
                x = F.pad(x, self._padding2)                
                x = super().forward(x)  
                return x             
            else: 
                t_c = self.t_c   
                b_t = t // t_c 
                # 将时间维度拆分为 b_t 个分组，每组 t_c 帧
                x = rearrange(x, 'b c (b_t t_c) h w -> (b b_t) c t_c h w', t_c=t_c)
                x = F.pad(x, self._padding)
                x = rearrange(x, '(b b_t) c t_c h w -> b b_t c t_c h w', b_t=b_t)
                # 对每个分组，第一帧使用 cache_x（或填充后的 cache_x）进行补全
                for i in range(b_t):
                    if i != 0: 
                        x[:, i, :, :1, :, :] = x[:, i-1, :, -2:-1, :, :]
                    else:
                        assert cache_x is not None, "t > 1 时，必须提供 cache_x 用于第一组的补全"
                        cache_x = cache_x.to(x.device)   
                        x[:, i, :, :1, :, :] = F.pad(cache_x, self._padding1)
                x = rearrange(x, 'b b_t c t_c h w -> (b b_t) c t_c h w')
                x = super().forward(x)  
                x = rearrange(x, '(b b_t) c t_c h w -> b c (b_t t_c) h w', b_t=b_t)  
                return x


if __name__ == '__main__':
    # 测试1：T == 1
    B, C, T, H, W = 1, 3, 1, 16, 16
    kernel = (3, 3, 3)
    stride = 1
    padding = (1, 1, 1)
    t_c = 2  # 时间维度分组因子

    model = Group_Causal_Conv3d(in_dim=C, out_dim=6, kernel=kernel, stride=stride, padding=padding, t_c=t_c)

    input = torch.randn(B, C, T, H, W)
    output = model(input)
    print(input.shape)
    print(output.shape)
