import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import *

from ..non_local_dot_product import NONLocalBlock2D


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, non_local_block=False):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1) / 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, bias=True)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, True)

        self.nl1 = NONLocalBlock2D(out_channels) if non_local_block else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1) / 2)),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, True)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.nl1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1) / 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, True)
    
    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1) / 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, True)

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1) / 2)),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, True)

    def forward(self, inp):
        x = self.bn0(inp)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


class SkipUNet3x3Up(nn.Module):

    def __init__(
            self, in_channel=8, out_channel=1, 
            num_channels_down=[], num_channels_up=[], num_channels_skip=[],
            kernel_sizes_down=[], kernel_sizes_up=[], kernel_sizes_skip=[],
            ):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ending = nn.Sequential(
            nn.Conv2d(num_channels_up[0], out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # down, skip
        in_channels = in_channel
        count = 0
        for num_channel_down, \
            kernel_size_down, \
            num_channel_skip, \
            kernel_size_skip in zip(num_channels_down, \
                                    kernel_sizes_down, \
                                    num_channels_skip, \
                                    kernel_sizes_skip):
            self.downs.append(DownSampleBlock(in_channels, num_channel_down, kernel_size_down, non_local_block=(count > 1)))
            self.skips.append(SkipBlock(in_channels, num_channel_skip, kernel_size_skip) if num_channel_skip != 0 else nn.Identity())
            in_channels = num_channel_down
            count += 1
        
        # up
        for n in range(len(num_channels_up)):
            num_channel_skip = num_channels_skip[-n-1]
            num_channel_up = num_channels_up[-n-1]
            kernel_size_up = kernel_sizes_up[-n-1]
            if num_channel_skip != 0:
                if n == 0:
                    self.ups.append(
                        nn.Sequential(
                            UpSampleBlock(num_channels_down[-1] + num_channels_skip[-1], num_channel_up, kernel_size_up),
                            nn.Upsample(scale_factor=2, mode='bilinear')
                        )
                    )
                elif n != len(num_channels_up) - 1:
                    self.ups.append(
                        nn.Sequential(
                            UpSampleBlock(num_channels_up[-n] + num_channels_skip[-n-1], num_channel_up, kernel_size_up),
                            nn.Upsample(scale_factor=2, mode='bilinear')
                        )
                    )
                else:
                    self.ups.append(
                        UpSampleBlock(num_channels_up[-n] + num_channels_skip[-n-1], num_channel_up, kernel_size_up)
                    )
            else:
                if n == 0:
                    self.ups.append(
                        nn.Sequential(
                            UpSampleBlock(num_channels_down[-1] + num_channels_down[-2], num_channel_up, kernel_size_up),
                            nn.Upsample(scale_factor=2, mode='bilinear')
                        )
                    )
                elif n != len(num_channels_up) - 1:
                    self.ups.append(
                        nn.Sequential(
                            UpSampleBlock(num_channels_up[-n] + num_channels_down[-n-2], num_channel_up, kernel_size_up),
                            nn.Upsample(scale_factor=2, mode='bilinear')
                        )
                    )
                else:
                    self.ups.append(
                        UpSampleBlock(num_channels_up[-n] + in_channel, num_channel_up, kernel_size_up)
                    )

    def forward(self, inp):
        
        x = inp
        _, _, H, W = x.shape

        skips = []
        skips.append(self.skips[0](x))

        for n, down in enumerate(self.downs):
            x = down(x)
            if n != len(self.downs) - 1:
                skips.append(self.skips[n+1](x))
        
        for n, (up, skip) in enumerate(zip(self.ups, skips[::-1])):
            if n == 0:
                x = self.upsample(x)
            x = self._concat((x, skip), dim=1)
            x = up(x)
            
        x = self.ending(x)
        return x[:, :, :H, :W]

    @staticmethod
    def _concat(inputs, dim):
        """
        inputs: 4-dim tensors
        dim: int.
        """
        heights = [inp.shape[2] for inp in inputs]
        widths = [inp.shape[3] for inp in inputs]

        if np.all(np.array(heights) == min(heights)) and np.all(np.array(widths) == min(widths)):
            inputs_ = inputs
        else:
            target_height = min(heights)
            target_width = min(widths)
            inputs_ = []
            for inp in inputs:
                diff_height = (inp.shape[2] - target_height) // 2
                diff_width = (inp.shape[3] - target_width) // 2
                inputs_.append(inp[:, :, diff_height: diff_height + target_height, diff_width: diff_width + target_width])
        
        return torch.cat(inputs_, dim=dim)
    
