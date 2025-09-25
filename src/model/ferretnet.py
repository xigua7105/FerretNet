from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0
    ):
        super().__init__()
        self.depth_wise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=padding, groups=in_channels, bias=False
        )
        self.point_wise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DilatedConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            dilation=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            groups=in_channels,
            dilation=2,
            bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            bias=False
        )

    def forward(self, x):
        x = self.conv1(x) + self.conv2(x)
        x = self.conv3(x)
        return x


class DSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int
    ):
        super().__init__()
        self.layers = nn.Sequential(
            DilatedConv2d(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layers(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class Ferret(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            dim: int,
            depths: list,
            lpd_func: str,
            window_size: int,
            lpd_dict: dict
    ):
        super().__init__()
        assert lpd_func in lpd_dict.keys()
        self.dim = dim

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim // 2),
            nn.ReLU(inplace=True)
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(self.dim // 2, self.dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True)
        )

        self.feature = nn.Sequential()
        for depth in depths:
            blocks = nn.Sequential(
                DSBlock(self.dim, self.dim * 2, stride=2),
                *[DSBlock(self.dim * 2, self.dim * 2, stride=1) for _ in range(depth - 1)]
            )
            self.feature.append(blocks)
            self.dim = self.dim * 2
        self.feature.append(
            nn.Conv2d(self.dim, self.dim, 1, 1, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(self.dim, num_classes)
        )

        self.lpd = lpd_dict[lpd_func](window_size)

    def forward(self, x):
        x = x - self.lpd(x)

        x = self.cbr1(x)
        x = self.cbr2(x)

        x = self.feature(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.logit(x)

        return x
