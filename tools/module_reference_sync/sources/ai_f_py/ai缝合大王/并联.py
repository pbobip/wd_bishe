# LKA
import torch
from einops import rearrange
from torch import nn


from 模块制作.LKA import LKA
# 并联 add
from 模块制作.串联 import AKConv

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out





class AKLKA1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lka = LKA(dim)
        self.ak = AKConv(inc=dim, outc=dim, num_param=3)
        self.mdta = Attention(64)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        res = x
        x1 = self.ak(x)
        x2 = self.lka(x)
        x3 = self.mdta(x)
        x = self.conv(x1 + x2 + x3)
        return x + res


# 并联 cat
class AKLKA2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lka = LKA(dim)
        self.ak = AKConv(inc=dim, outc=dim, num_param=3)
        self.mdta = Attention(64)
        self.conv = nn.Conv2d(dim*3, dim, 1)

    def forward(self, x):
        res = x
        x1 = self.ak(x)
        x2 = self.lka(x)
        x3 = self.mdta(x)
        x =torch.cat((x1,x2,x3),dim=1)
        x = self.conv(x)
        return x + res


# 并联 cat
class AKLKA3(nn.Module):
    def __init__(self, dim,branch_ratio=0.25):
        super().__init__()
        sp = int(dim * branch_ratio)
        self.lka = LKA(sp)
        self.ak = AKConv(inc=sp, outc=sp, num_param=3)
        self.mdta = Attention(dim-sp*2)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.split_indexes = (sp, sp,dim - 2 * sp)
    def forward(self, x):
        res = x

        x_ak, x_lka, x_mdta = torch.split(x, self.split_indexes, dim=1)
        x1 = self.ak(x_ak)
        x2 = self.lka(x_lka)
        x3 = self.mdta(x_mdta)
        x =torch.cat((x1,x2,x3),dim=1)

        x = self.conv(x)
        return x + res

# 输入 N C H W,  输出 N C H W

if __name__ == '__main__':
    block = AKLKA3(64)
    input = torch.rand(3, 64, 56, 56)
    output = block(input)
    print(input.size(), output.size())
