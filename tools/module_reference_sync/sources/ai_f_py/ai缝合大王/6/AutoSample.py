import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：AutoLUT: LUT-Based Image Super-Resolution with Automatic Sampling and Adaptive Residual Learning
class AutoSample(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_shape=input_size
        self.sampler=nn.Conv2d(1,4,input_size)
        self.shuffel=nn.PixelShuffle(2)
        self.nw=input_size**2

    def forward(self, x):
        assert len(x.shape)==4 and x.shape[-2:]==(self.input_shape,self.input_shape), f"Unexpected shape: {x.shape}"
        # x = self.sampler(x)
        # logger.debug(self.sampler.weight)
        w = F.softmax(self.sampler.weight.view(-1, self.nw), dim=1).view_as(self.sampler.weight)
        x = F.conv2d(x, w, bias=self.sampler.bias*0)
        x = self.shuffel(x)
        return x


if __name__ == '__main__':
    # B=1, C=1, H=8, W=8
    B, C, H, W = 1, 1, 8, 8
    model = AutoSample(input_size=H)

    input = torch.randn(B, C, H, W)
    output = model(input)

    print("input size:", input.size())
    print("output size:", output.size())
