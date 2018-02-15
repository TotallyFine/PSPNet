# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from .ResNet34 import resnet34 # the function to load pretrained resnet34
from .BasicModule import BasicModule

# Pyramid Pooling Module
class PPModule(nn.Module):
    def __init__(self, in_c, out_c=1024, sizes=(1, 2, 3, 6)):
        '''
        in_c is input channel
        out_c is output channel
        sizes is the feature's size which in Pyramid Pooling Module
        small size panel used to classify and bigger size panel are better in detecting the border of object
        in Pyramid Pooling Module all conv's kernel size is 1, that's for preserve overall situation's info and reduce conv operation's enpenditures
        '''
        super(PSPModule, self).__init__()
        self.panels = nn.ModuleList([self._make_panel(in_c, size) for size in sizes])
        # bottleneck combines panels and input
        self.bottleneck = nn.Conv2d(in_c * (len(sizes) + 1), out_c, kernel_size=1)
        self.relu = nn.ReLU()
        
    def _make_panel(self, in_c, size):
        # nn.AdaptiveAvgPool2d is a self-adapte pool layer,it takes different size features but return same size doesn't change channel num
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
        
    def forward(self, feat):
        # feat (batch_size, channel=512, h=height/4, w=width/4)
        h, w = feat.size(2), feat.size(3)
        
        # use bilinear interpolation
        # prior = [p1, p2, p3, p6, feat]
        prior = [F.upsample(input=panel(feat), size(h, w), mode='bilinear') for panel in self.panels] + [feat]
        # concat to (batch_size, channel, h, w) and using conv to combine
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
        
# upsample 2×
class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.COnv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.PReLU()
        )
        
    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)
        
class PSPNet(BasicModule):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), pretrained=True):
        super(PSPNet, self).__init__()
        self.model_name = 'PSPNet'
        # feature extracting
        self.feat = resnet34(pretrained)
        # resnet34 output's channel is 512
        self.ppm = PPModule(512, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        
        self.up_1 = UpSample(1024, 256)
        self.up_2 = UpSample(256, 64)
        self.up_3 = UpSample(64, 64)
        
        self.drop_2 = nn.Dropout2d(p=0.15)
        
        # create the mask 
        self.mask = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )
        # resnet34's x_3
        #self.classifier = nn.Sequential(
        #    nn.Linear(512, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, n_classes)
        #)
        
    def forward(self, x):
        f = self.feat(x) #h/4 w/4
        p = self.ppm(f)
        p = self.drop_1(p)
        
        p = self.up_1(p)
        p = self.drop_2(p)
        
        p = self.up_2(p)
        p = self.drop_2(p)
        
        p = self.up_3(p)
        p = self.drop_2(p)
        
        return self.mask(p)     
