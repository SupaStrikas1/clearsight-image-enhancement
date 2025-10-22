import torch
import torch.nn as nn

class MCSA(nn.Module):
    def __init__(self, channels):
        super(MCSA, self).__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = qkv
        attn = self.attn(q @ k.transpose(-2, -1))
        return attn @ v + x

class GCFN(nn.Module):
    def __init__(self, channels):
        super(GCFN, self).__init__()
        self.pw1 = nn.Conv2d(channels, channels * 4, 1)
        self.dw = nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4)
        self.gelu = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x):
        return self.pw2(self.gelu(self.dw(self.pw1(x)))) + x

class MulSormer(nn.Module):
    def __init__(self, base_channels=16, levels=4):
        super(MulSormer, self).__init__()
        self.base_channels = base_channels
        self.levels = levels
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    base_channels * (2**(i-1)) if i > 0 else 3,
                    base_channels * (2**i),
                    3, stride=2, padding=1
                ),
                MCSA(base_channels * (2**i)),
                GCFN(base_channels * (2**i))
            ) for i in range(levels)
        ])
        self.bottleneck = nn.Sequential(
            MCSA(base_channels * (2**(levels-1))),
            GCFN(base_channels * (2**(levels-1)))
        )
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    base_channels * (2**(levels-1)) if i == levels-1 else base_channels * (2**(i+1)),
                    base_channels * (2**i),
                    3, stride=2, padding=1, output_padding=1
                ),
                nn.ReLU()
            ) for i in range(levels-1, -1, -1)
        ])
        self.final_conv = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            skips.append(x)
            print(f"Encoder level {i}: {x.shape}")
        x = self.bottleneck(x)
        print(f"Bottleneck: {x.shape}")
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip = skips.pop()
            print(f"Decoder level {i}: x={x.shape}, skip={skip.shape}")
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        x = self.final_conv(x)
        # Ensure final output is 256x256
        if x.shape[2:] != (256, 256):
            x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        print(f"Final output: {x.shape}")
        return x

if __name__ == "__main__":
    model = MulSormer(base_channels=16)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)  # Expected: torch.Size([1, 3, 256, 256])