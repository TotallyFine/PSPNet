# coding:utf-8
from collections import OrderDict

from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo

# resnet 34
class ResidualBlock(nn.Module):
    '''
    using dilation conv
    '''
    def __init__(self, in_c, out_c, stride=1, shortcut=None, dilation=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=Ture),
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.right = shortcut
        
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
        
class ResNet34(nn.Module):
    '''
    ResNet34 used for feature extractors.
    output is x, x_3.the size of them are (batch_size, 512, h/4, w/4)
    '''
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'
        
        
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), # h/2 w/2
            nn.BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1) # h/4 w/4
        )

       
        self.layer1 = self._make_layer(64, 128, 3, stride=1)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=1, dilation=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=1, dilation=4)


    def _make_layer(inchannel, outchannel, block_num, stride=1, dilation=1):
        
        shortcut = nn.Sequential(
                       nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                       nn.BatchNorm2d(outchannel)
                   )
        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut, dilation))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x

def resnet34(pretrained=True):
    '''
    load pretrained or no trained resnet34 module
    '''
    model = ResNet34() 
    if pretrained:
        source = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        new_dict = OrderDict()
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), source.items()):
            new_dict[k1] = v2
        model.load_state_dict(new_dict)
    return model
