import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import *

###############################################################################

# 기존 NAFNet-DIP와의 차이점
## 1. Downsample을 NAFBlock 이후 2x2 conv로 수행하지 않고, NAFBlock 내의 3x3 dconv
##### 를 3x3 conv with stride 2로 바꾸어 수행
## 2. skip connection을 그냥 residual connection대신 bn-1x1 conv-SG로 수행
## 3. Upsample 위치를 첫번째 bn 이후로 바꿈

###############################################################################


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.sg = SimpleGate()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=3, stride=2, padding=0, bias=True)
        )
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inp):
        x = self.bn1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.bn2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        return x


class NAFSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.sg = SimpleGate()
    
    def forward(self, inp):
        x = self.bn1(inp)
        x = self.conv1(x)
        x = self.sg(x)
        return x


class NAFUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sg = SimpleGate()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        )
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inp):
        x = self.bn1(inp)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.bn2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        
        return x


class NAFNet_DIPv2(nn.Module):

    def __init__(
            self, in_channel, out_channel, 
            num_channels_down=[], num_channels_up=[], num_channels_skip=[]
            ):
        super().__init__()

        self.downs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.ending = nn.Sequential(
            nn.Conv2d(num_channels_up[0], out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # down, skip
        in_channels = in_channel
        for num_channel_down, num_channel_skip in zip(num_channels_down, num_channels_skip):
            self.downs.append(NAFDownBlock(in_channels, num_channel_down))
            in_channels = num_channel_down
            self.skips.append(NAFSkipBlock(in_channels, num_channel_skip))
        
        # up
        in_channels = num_channel_skip
        for num_channel_up in num_channels_up[::-1]:
            self.ups.append(NAFUpBlock(in_channels, num_channel_up))
            in_channels = num_channel_up
    
    def forward(self, inp):

        x = inp
        _, _, H, W = x.shape

        after_downs = []

        for down in self.downs:
            x = down(x)
            after_downs.append(x)
        
        for n in range(len(self.ups)):
            up = self.ups[n]
            skip = self.skips[-n-1]
            after_down = after_downs[-n-1]

            if n == 0:
                x = skip(x)
                x = up(x)
            else:
                x = self._add_features((x, skip(after_down)))
                x = up(x)
        
        x = self.ending(x)
        return x[:, :, :H, :W]
    
    @staticmethod
    def _add_features(inputs):
        """
        inputs: 4-dim tensors
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
        
        return torch.sum(torch.stack(inputs_), dim=0)
