import torch
import torch.nn as nn
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # compute Z[L+2]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        # downsample a[L] in case there is stride in conv1
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x.clone()
        if self.downsample:
            identity = self.downsample(identity)
        return self.relu(self.conv(x) + identity)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, x.size()[2:])


class NaiveResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.groups = nn.ModuleList([
            self._build_group(in_channels=3, out_channels=64, stride=2, num_blocks=2),
            self._build_group(in_channels=64, out_channels=128, stride=2, num_blocks=2),
            self._build_group(in_channels=128, out_channels=512, stride=2, num_blocks=2),
            self._build_group(in_channels=512, out_channels=1024, stride=2, num_blocks=2)
        ])
        self.globalavgpool = GlobalAveragePooling()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=200, kernel_size=1)
        )

    def forward(self, x):
        for group in self.groups:
            x = group(x)
        # global average pooling
        x = self.globalavgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

    def _build_group(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)


if __name__ == '__main__':
    dummy_input = Variable(torch.rand(16, 3, 64, 64))
    resnet = NaiveResNet(num_classes=200)
    y = resnet.forward(dummy_input)
    print(y.size())