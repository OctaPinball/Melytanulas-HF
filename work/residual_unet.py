import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x + shortcut)  # Residual connection

class ResidualUNet(nn.Module):
    def __init__(self):
        super(ResidualUNet, self).__init__()

        # Encoder
        self.enc1 = ResidualBlock(1, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))
        self.enc5 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024))

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            ResidualBlock(512, 512)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ResidualBlock(256, 256)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ResidualBlock(64, 64)
        )

        # Final output layer
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        # Decoder
        dec4_out = self.dec4(enc5_out) + enc4_out  # Skip connection
        dec3_out = self.dec3(dec4_out) + enc3_out  # Skip connection
        dec2_out = self.dec2(dec3_out) + enc2_out  # Skip connection
        dec1_out = self.dec1(dec2_out) + enc1_out  # Skip connection

        return self.final(dec1_out)
