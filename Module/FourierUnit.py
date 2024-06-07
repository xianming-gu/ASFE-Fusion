import torch
import torch.nn as nn
from torch.fft import irfft2
from torch.fft import rfft2


def rfft(x, d):
    t = rfft2(x, dim=(-d, -1))
    return torch.stack((t.real, t.imag), -1)


def irfft(x, d, signal_sizes):
    return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d, -1))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaFreFusion(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(AdaFreFusion, self).__init__()

        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv_layer1 = torch.nn.Conv2d(in_channels=2, out_channels=1,
                                           kernel_size=7, stride=1, padding=7 // 2, groups=self.groups, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = rfft(x, 2)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        X_F = ffted.view((batch, -1,) + ffted.size()[3:])
        X_F = self.conv_layer(X_F)  # (batch, c*2, h, w/2+1)
        X_F = self.relu(self.bn(X_F))

        avg_out = torch.mean(X_F, dim=1, keepdim=True)
        max_out, _ = torch.max(X_F, dim=1, keepdim=True)
        poolre = torch.cat([avg_out, max_out], dim=1)
        ffted1 = self.conv_layer1(poolre)
        spatial_mask = self.sigmoid(ffted1)

        X_F_hat = X_F * spatial_mask
        X_F_hat = X_F_hat.view((batch, -1, 2,) + X_F_hat.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        output = irfft(X_F_hat, 2, signal_sizes=r_size[2:])

        return output
