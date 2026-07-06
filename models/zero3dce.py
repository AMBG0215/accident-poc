# Zero-3DCE architecture — must match Zero3DCE_Train.ipynb exactly
# (copied verbatim from the thesis training notebook).
import torch
import torch.nn as nn


class SepConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = SepConv3d(in_ch, out_ch)
        self.attn = SpatialAttention()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.attn(self.conv(x))
        return self.pool(x), x


class Zero3DCE(nn.Module):
    def __init__(self, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 32)
        self.enc3 = EncoderBlock(32, 32)
        self.bottleneck = SepConv3d(32, 32)
        self.dec1 = SepConv3d(64, 32)
        self.dec2 = SepConv3d(64, 32)
        self.dec3 = SepConv3d(64, 24)
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.tanh = nn.Tanh()

    def apply_curve(self, frame, alpha):
        return torch.clamp(frame + alpha * frame * (frame - 1), 0, 1)

    def forward_with_alphas(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        xb = self.bottleneck(x3)
        xb = self.dec1(torch.cat([self.up(xb), skip3], dim=1))
        xb = self.dec2(torch.cat([self.up(xb), skip2], dim=1))
        xb = self.dec3(torch.cat([self.up(xb), skip1], dim=1))
        alpha_maps = self.tanh(xb)
        enhanced = x
        for i in range(self.num_iterations):
            g = (i % 3) * 8
            enhanced = self.apply_curve(enhanced, alpha_maps[:, g:g + 3])
        return enhanced, alpha_maps
