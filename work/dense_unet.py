import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
    
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(outputs, dim=1))
            outputs.append(out)
        return torch.cat(outputs, dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)

class DenseUNet(nn.Module):
    def __init__(self, input_channels=1, growth_rate=32, num_layers_per_block=4):
        super(DenseUNet, self).__init__()
        self.init_conv = nn.Conv2d(input_channels, growth_rate * 2, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder_block1 = DenseBlock(growth_rate * 2, growth_rate, num_layers_per_block)
        self.trans1 = TransitionLayer(growth_rate * (2 + num_layers_per_block), growth_rate * 2)
        
        self.encoder_block2 = DenseBlock(growth_rate * 2, growth_rate, num_layers_per_block)
        self.trans2 = TransitionLayer(growth_rate * (2 + num_layers_per_block), growth_rate * 2)
        
        self.bottleneck = DenseBlock(growth_rate * 2, growth_rate, num_layers_per_block)
        
        self.upconv1 = nn.ConvTranspose2d(growth_rate * (2 + num_layers_per_block), growth_rate * 2, kernel_size=2, stride=2)
        self.decoder_block1 = DenseBlock(growth_rate * 4, growth_rate, num_layers_per_block)
        
        self.upconv2 = nn.ConvTranspose2d(growth_rate * (2 + num_layers_per_block), growth_rate * 2, kernel_size=2, stride=2)
        self.decoder_block2 = DenseBlock(growth_rate * 4, growth_rate, num_layers_per_block)
        
        self.final_conv = nn.Conv2d(growth_rate * (2 + num_layers_per_block), 2, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)
        
        enc1 = self.encoder_block1(x)
        x = self.trans1(enc1)
        
        enc2 = self.encoder_block2(x)
        x = self.trans2(enc2)
        
        x = self.bottleneck(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder_block1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder_block2(x)
        
        return self.final_conv(x)
