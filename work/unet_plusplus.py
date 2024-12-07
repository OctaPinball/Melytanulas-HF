import torch
import torch.nn as nn
import torch.nn.functional as F

class NestedConvBlock(nn.Module):
    """A nested block in U-Net++ with dense skip connections."""
    def __init__(self, in_channels, out_channels):
        super(NestedConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x)

class UNetPlusPlus(nn.Module):
    def __init__(self):
        super(UNetPlusPlus, self).__init__()

        # Number of features for each layer
        features = [64, 128, 256, 512, 1024]

        # Encoder (Downsampling Path)
        self.enc1 = NestedConvBlock(1, features[0])
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), NestedConvBlock(features[0], features[1]))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), NestedConvBlock(features[1], features[2]))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), NestedConvBlock(features[2], features[3]))
        self.enc5 = nn.Sequential(nn.MaxPool2d(2), NestedConvBlock(features[3], features[4]))

        # Decoder (Upsampling Path with Nested Connections)
        self.up4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.dec4 = NestedConvBlock(features[3] + features[3], features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = NestedConvBlock(features[2] + features[2] + features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = NestedConvBlock(features[1] + features[1] + features[2] + features[3], features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = NestedConvBlock(features[0] + features[0] + features[1] + features[2] + features[3], features[0])

        # Final Output Layer
        self.final = nn.Conv2d(features[0], 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        # Decoder with Nested Connections
        up4_out = self.up4(enc5_out)
        dec4_out = self.dec4(torch.cat([up4_out, enc4_out], dim=1))

        up3_out = self.up3(dec4_out)
        dec3_out = self.dec3(torch.cat([up3_out, enc3_out, dec4_out], dim=1))

        up2_out = self.up2(dec3_out)
        dec2_out = self.dec2(torch.cat([up2_out, enc2_out, dec3_out, dec4_out], dim=1))

        up1_out = self.up1(dec2_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc1_out, dec2_out, dec3_out, dec4_out], dim=1))

        # Final Output
        return self.final(dec1_out)
