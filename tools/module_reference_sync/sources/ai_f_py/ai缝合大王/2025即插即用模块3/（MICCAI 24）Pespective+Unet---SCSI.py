import torch
import torch.nn as nn
from einops import rearrange

class SCSI(nn.Module):
    def __init__(self, in_channels, inter_dim=256, h_w=[(56, 56), (28, 28), (14, 14)]):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, inter_dim, 1) for c in in_channels])
        self.reprojs = nn.ModuleList([nn.Conv2d(inter_dim, c, 1) for c in in_channels])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=inter_dim, nhead=8), num_layers=3
        )
        self.h_w = h_w

    def forward(self, inputs):
        flat = []
        for i, x in enumerate(inputs):
            x_proj = self.projs[i](x)
            flat.append(rearrange(x_proj, 'b c h w -> b (h w) c'))
        x_cat = torch.cat(flat, dim=1)
        x_trans = self.transformer(x_cat)
        splits = torch.split(x_trans, [h * w for h, w in self.h_w], dim=1)
        out = []
        for i, s in enumerate(splits):
            x = rearrange(s, 'b (h w) c -> b c h w', h=self.h_w[i][0], w=self.h_w[i][1])
            out.append(self.reprojs[i](x))
        return out
# ...（上方 SCSI 定义保持不变）

if __name__ == '__main__':
    x1 = torch.randn(4, 256, 56, 56)
    x2 = torch.randn(4, 512, 28, 28)
    x3 = torch.randn(4, 1024, 14, 14)
    print("Input shapes:", x1.shape, x2.shape, x3.shape)
    model = SCSI(in_channels=[256, 512, 1024])
    outputs = model([x1, x2, x3])
    for i, out in enumerate(outputs):
        print(f"Output[{i}] shape:", out.shape)
