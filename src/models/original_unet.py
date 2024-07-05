import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     DoubleConv(in_channels, out_channels)
        # )
        self.strided_conv = nn.Sequential(
            Convolution(spatial_dims=3, in_channels=in_channels, out_channels=in_channels, kernel_size=3, strides=2, padding=1, adn_ordering="NDA", norm="batch", act='relu', bias=False),
            DoubleConv(in_channels, out_channels)
        )
        

    def forward(self, x):
        return self.strided_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # # input is BCHWD
        # diffH = x2.size()[2] - x1.size()[2]
        # diffW = x2.size()[3] - x1.size()[3]
        # diffD = x2.size()[4] - x1.size()[4]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class UNetRon(nn.Module):
    def __init__(self, n_channels, n_classes, features = 64, trilinear=False):
        super(UNetRon, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = (DoubleConv(n_channels, features))
        self.down1 = (Down(features, features*2))
        self.down2 = (Down(features*2, features*4))
        self.down3 = (Down(features*4, features*8))
        factor = 2 if trilinear else 1
        self.down4 = (Down(features*8, features*16 // factor))
        self.up1 = (Up(features*16, features*8 // factor, trilinear))
        self.up2 = (Up(features*8, features*4 // factor, trilinear))
        self.up3 = (Up(features*4, features*2 // factor, trilinear))
        self.up4 = (Up(features*2, features, trilinear))
        self.outc = (OutConv(features, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
if __name__ == "__main__":
    import time 
    import numpy as np

    model = UNet(n_channels=1, n_classes=1, trilinear=False) 
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable paramters: {pytorch_trainable_params}, all parameters: {pytorch_total_params}.")
    # input = torch.rand(1, 1, 128, 128, 128)
    # tab = torch.rand(1, 544)
    # output = model(input, tab)
    # print(output.shape)
    
    device = 'cuda:1'
    img = torch.rand(1, 1, 224, 224, 160).to(device)
    model = model.to(device)
    time_list = []
    for i in range(30):
        start = time.time()
        output = model(img)
        torch.cuda.synchronize(device)
        end = time.time()-start
        if i > 10:
            time_list.append(end)
    print(output.shape)
    print(f"avg. forward time: {np.array(time_list).mean()*1000:.3f}ms.")
    
    
