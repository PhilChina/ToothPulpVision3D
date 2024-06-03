import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2), pool_stride=(2, 2)):
        super(ConvEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(2, 2), stride=(2, 2), bias=True)
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        up_height, up_width = x.size()[2:]
        sk_height, sk_width = skip.size()[2:]

        diff_y = sk_height - up_height
        diff_x = sk_width - up_width

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


def test_ConvDecoder():
    x = torch.rand([1, 128, 8, 8])
    skip = torch.rand([1, 64, 16, 17])

    net = ConvDecoder(in_channels=128, out_channels=64)
    x = net(x, skip)
    print(x.size())


if __name__ == '__main__':
    test_ConvDecoder()
