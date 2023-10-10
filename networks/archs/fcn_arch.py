import torch
import torch.nn as nn
from ..common import *


class FCN(nn.Module):
    def __init__(self, num_input_channels=200, num_output_channels=1, num_hidden=1000):
        super().__init__()
        self.linear1 = nn.Linear(num_input_channels, num_hidden, bias=True)
        self.act1 = nn.ReLU6()
        self.linear2 = nn.Linear(num_hidden, num_output_channels)
        self.act2 = nn.Softmax()
    
    def forward(self, inp):
        x = self.linear1(inp)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x
