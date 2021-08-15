import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def init_weights(net, init_type = None, gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) & (hasattr(m, 'weight')):
            if init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = gain)
            elif init_type == 'normal':
                init.normal_(m.weight.data, mean = 0.0, std = gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = gain)
            if hasattr(m, 'bias') and (m.bias is not None):
                init.constant_(m.bias.data, val = 0.0)
        # Batch 정규화를 진행한다면 각각의 scalar feature들을 독립적으로 정규화하게 된다.
        # 즉, 각각의 feature들의 평균과 분산을 0과 1로 정규화 함으로서 
        # 각 차원들의 activation 각각에 대해서 수행되는 per-dimension variance를 계산함
    
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, mean = 1.0, std = gain)
            init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = ch_out, out_channels = ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels = ch_in, out_channels = ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        return self.conv(x)

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)