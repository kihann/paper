import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from data import get_stl10_data
from noise import NoiseScheduler, device

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, dim]
        return emb

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        num_groups = min(32, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)

    def forward(self, x, t_emb):
        h = self.norm1(self.block1(x))
        h += self.time_proj(t_emb)[:, :, None, None]
        h = F.relu(h)
        h = self.norm2(self.block2(h))
        h = F.relu(h)
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=256):
        super().__init__()
        channel_sizes = [in_channels, 64, 128, 256, 512, 512]
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(len(channel_sizes) - 1):
            self.down_blocks.append(UNetBlock(channel_sizes[i], channel_sizes[i + 1], time_emb_dim))

        self.mid = UNetBlock(channel_sizes[-1], channel_sizes[-1], time_emb_dim)
        
        for i in reversed(range(len(channel_sizes) - 1)):
            self.up_blocks.append(UNetBlock(channel_sizes[i + 1] * 2, channel_sizes[i], time_emb_dim))

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out = nn.Conv2d(channel_sizes[0], in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        downs = []
        for down in self.down_blocks:
            x = down(x, t_emb)
            downs.append(x)
            x = self.pool(x)

        mid = self.mid(x, t_emb)

        for up in self.up_blocks:
            x = self.up(mid)
            x = torch.cat([x, downs.pop()], dim=1)
            mid = up(x, t_emb)

        return self.out(mid)
