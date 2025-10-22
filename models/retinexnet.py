import torch
import torch.nn as nn

class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 6, 3, padding=1),  # Reflectance + Illumination
            nn.Sigmoid()
        )

    def forward(self, x):
        r_i = self.conv(x)
        r, i = r_i[:, :3], r_i[:, 3:]
        return r, i

class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, i):
        return self.decoder(self.encoder(i))

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.decom = DecomNet()
        self.enhance = EnhanceNet()

    def forward(self, x):
        r, i = self.decom(x)
        i_enhanced = self.enhance(i)
        return r * i_enhanced  # Reconstruct

if __name__ == "__main__":
    model = RetinexNet()
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)  # Expected: torch.Size([1, 3, 256, 256])