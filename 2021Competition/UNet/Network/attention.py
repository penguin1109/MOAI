import torch
import torch.nn as nn
import torch.functional as F

class Attention_block(nn.Module):
    # features for gating vector = F_g
    # features for pixel vector = F_l
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(num_features = F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(num_features = F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, sride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(num_features = 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x, g):
        g1 = self.W_g(x)
        x1 = self.W_x(x)
        new = g1 + x1
        psi = self.psi(self.relu(new))

        return x*psi # 순서 바뀌면 절대 안됨

    


