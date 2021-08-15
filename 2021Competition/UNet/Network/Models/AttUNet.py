import torch
import torch.nn as nn
from conv import up_conv
from attention import Attention_block
from conv import conv_block

class AttUNet(nn.Module):
    # filtersize 3 2배로 upsampling ReLU activation function
    def __init__(self, first_output_ch, img_ch = 3, fin_output_ch = 1):
        super(AttUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = nn.Conv2d(in_channels = img_ch, out_channels = 64)
        self.Conv2 = nn.Conv2d(in_channels = 64, out_channels = 128)
        self.Conv3 = nn.Conv2d(in_channels = 128, out_channels = 256)
        self.Conv4 = nn.Conv2d(in_channels = 256, out_channels = 512)
        self.Conv5 = nn.Conv2d(in_channels = 512, out_channels = 1024)

        self.Up5 = up_conv(in_channels = 1024, out_channels = 512)
        self.Att5 = Attention_block(F_g = 512, F_l = 512, F_int = 256)
        self.Up_conv5 = conv_block(ch_in = 1024, ch_out = 512)

        self.Up4 = up_conv(in_channels = 512, out_channels = 256)
        self.Att4 = Attention_block(F_g = 256, F_l = 256, F_int = 128)
        self.Up_conv4 = conv_block(ch_in = 512, ch_out = 256)

        self.Up3 = up_conv(in_channels = 256, out_channels = 128)
        self.Att3 = Attention_block(F_g = 128, F_l = 128, F_int = 64)
        self.Up_conv3 = conv_block(ch_in = 256, ch_out = 128)

        self.Up2 = up_conv(in_channels = 128, out_channels = 128)
        self.Att2 = Attention_block(F_g = 64, F_l = 64, F_int = 32)
        self.Up_conv2 = conv_block(ch_in = 128, ch_out = 64)

        self.Conv1x1 = nn.Conv2d(64, fin_output_ch, kernel_size = 1, stride = 1, padding = 1, bias = True)

    def forward(self, x):
        # Encoding 
        x1 = self.Conv1(x)

        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)

        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # Decoding + Concatenation
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
