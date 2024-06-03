import torch
import torch.nn as nn

from model.module import ConvEncoder, ConvDecoder


def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net


class Unet(nn.Module):
    def __init__(self, in_channel=None, out_channel=None, mid_channels=None):
        super(Unet, self).__init__()

        if mid_channels is None:
            mid_channels = [32, 64, 128, 256]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mid_channels = mid_channels
        self.num_layers = len(self.mid_channels) - 1

        self.incoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.mid_channels[0] // 2,
                      kernel_size=(3, 3),
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=self.mid_channels[0] // 2),
            nn.LeakyReLU(inplace=True)
        )

        self.encoders = nn.ModuleList()
        for index, channel in enumerate(self.mid_channels):
            self.encoders.append(
                ConvEncoder(in_channels=self.mid_channels[index - 1] if index > 0 else self.mid_channels[0] // 2,
                            out_channels=channel)
            )

        self.decoders = nn.ModuleList()
        for index, channel in enumerate(self.mid_channels):
            index = self.num_layers - index
            self.decoders.append(
                ConvDecoder(in_channels=self.mid_channels[index],
                            out_channels=self.mid_channels[index - 1] if index > 0 else self.mid_channels[0] // 2)
            )

        self.outcoder = nn.Sequential(
            nn.Conv2d(in_channels=self.mid_channels[0] // 2,
                      out_channels=self.out_channel,
                      kernel_size=(1, 1),
                      stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, image):
        x = self.incoder(image)
        skipes = [x]
        for index, encoder in enumerate(self.encoders):
            skipes.append(encoder(skipes[index]))

        for index, decoder in enumerate(self.decoders):
            x = decoder(x=x if index > 0 else skipes[-1], skip=skipes[-index - 2])

        x = self.outcoder(x)
        return x


def test_Unet():
    x = torch.rand([1, 1, 256, 128]).cuda()
    net = Unet(in_channel=1, out_channel=1, mid_channels=[64, 128, 256, 512]).cuda()
    x = net(x)
    print(x.size())


if __name__ == "__main__":
    test_Unet()
