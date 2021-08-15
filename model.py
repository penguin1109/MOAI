import torch
import torch.nn as nn
import torchvision.transforms.functional as VF
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(num_features = ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, ch_in, ch_out, features = [32, 64, 128, 256]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        for feature in features:
            self.downs.append(DoubleConv(ch_in, feature))
            ch_in = feature
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size = 2, stride = 2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # final convolution layer이 필요한 이유는 마지막에 Unet 구조를 통해서 얻은 feature들을 바탕으로 
        # 3개의 class(0, 1, 2)로 나누어 주어야 하기 때문이다.
        self.final_conv = nn.Conv2d(features[0], ch_out, kernel_size = 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = VF.resize(x, size = skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim = 1) # channel에 대해서 concatenation을 진행
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((1, 3, 512, 512))
    model = UNET(ch_in = 3, ch_out = 3)
    preds = model(x)
    print(preds)
    #print(preds.shape)
    #plt.imshow(preds[0].permute(1, 2, 0).detach().numpy(), cmap = 'gray')
    #plt.show()
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()