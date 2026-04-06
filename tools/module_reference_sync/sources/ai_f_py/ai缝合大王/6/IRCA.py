import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction

#CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution


def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))
    
    
def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
      
    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k,v, x_global.detach()

if __name__ == '__main__':
    # B: batch_size, N: 序列长度, C: 特征维度
    B, N, C = 1, 16, 64
    qk_dim, heads = 32, 4

    input_norm = torch.randn(B, N, C)
    input_means = torch.randn(B, N, C)

    model = IRCA(dim=C, qk_dim=qk_dim, heads=heads)
    model.eval()
    with torch.no_grad():
        k, v, x_global = model(input_norm[0], input_means[0])

    print("k:", k.shape)         # (heads, N, qk_dim/heads)
    print("v:", v.shape)         # (heads, N, C/heads)
    print("x_global:", x_global.shape)  # (N, C)

