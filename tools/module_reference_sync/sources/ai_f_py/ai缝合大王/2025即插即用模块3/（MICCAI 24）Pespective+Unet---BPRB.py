import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.relu(self.bn(self.conv(x)))

class BiPathResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilation=2, use_dilate_conv=True):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            ResBlock(mid_channels, mid_channels)
        )
        self.dilated_resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            DilatedResBlock(mid_channels, mid_channels, dilation=dilation)
        )
        self.fusionblock = FusionBlock(2 * mid_channels, out_channels)
        self.use_dilate_conv = use_dilate_conv

    def forward(self, x):
        res_out = self.resblock(x)
        dilated_out = self.dilated_resblock(x)
        if self.use_dilate_conv:
            return self.fusionblock(res_out, dilated_out)
        else:
            return self.fusionblock(res_out, res_out)


if __name__ == '__main__':
    input = torch.randn(4, 3, 128, 128)
    print("Input shape:", input.size())
    block = BiPathResBlock(in_channels=3, mid_channels=32, out_channels=64, dilation=2)
    output = block(input)
    print("Output shape:", output.size())
