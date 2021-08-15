# implementation of convolutional neural network implemented with recurrent blocks

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils import data

class Recurrent_block(nn.Module):
    def __init__(self, ch_in, ch_out, t = 2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = ch_in, out_channels = ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t = 2):
        super(RRCNN_block, self).__init__()
        self.t = t
        self.RCNN = nn.Sequential(
            nn.Conv2d(in_channels = ch_in, out_channels = ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 1, bias = True)
    
    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1