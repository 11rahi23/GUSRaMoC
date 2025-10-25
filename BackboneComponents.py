import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv(nn.Module):
    """Standard convolution with batch norm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 3, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """C2f module from YOLOv8"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.conv2 = Conv((2 + n) * self.hidden_channels, out_channels, 1, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, groups, 1.0) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class YOLOv8Backbone(nn.Module):
    """YOLOv8 Backbone for feature extraction"""
    def __init__(self, in_channels=3, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        
        # Stem
        self.stem = Conv(in_channels, int(64 * width_mult), 3, 2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(int(64 * width_mult), int(128 * width_mult), 3, 2),
            C2f(int(128 * width_mult), int(128 * width_mult), int(3 * depth_mult), True)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(int(128 * width_mult), int(256 * width_mult), 3, 2),
            C2f(int(256 * width_mult), int(256 * width_mult), int(6 * depth_mult), True)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(int(256 * width_mult), int(512 * width_mult), 3, 2),
            C2f(int(512 * width_mult), int(512 * width_mult), int(6 * depth_mult), True)
        )
        
        # Stage 4
        # Edited following YOLOv8 paper
        self.stage4 = nn.Sequential(
            Conv(int(512 * width_mult), int(1024 * width_mult), 3, 2),
            C2f(int(1024 * width_mult), int(1024 * width_mult), int(3 * depth_mult), True),
            SPPF(int(1024 * width_mult), int(1024 * width_mult), 5)
            #Conv(int(512 * width_mult), int(512 * width_mult), 3, 2),
            #C2f(int(512 * width_mult), int(512 * width_mult), int(3 * depth_mult), True),
            #SPPF(int(512 * width_mult), int(512 * width_mult), 5)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)  # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32
        return p3, p4, p5


class YOLOv8Neck(nn.Module):
    """YOLOv8 Neck (PANet)"""
    def __init__(self):
        super().__init__()
        
        # Top-down FPN
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fpn_p4 = C2f(512 + 1024, 512, 3, False)
        #self.fpn_p4 = C2f(512 + 512, 512, 3, False)
        self.fpn_p3 = C2f(256 + 512, 256, 3, False)
        
        # Bottom-up PAN
        self.downsample_p3 = Conv(256, 256, 3, 2)
        self.pan_p4 = C2f(256 + 512, 512, 3, False)
        self.downsample_p4 = Conv(512, 512, 3, 2)
        self.pan_p5 = C2f(512 + 1024, 1024, 3, False)
        #self.pan_p5 = C2f(512 + 512, 1024, 3, False)

    def forward(self, features):
        p3, p4, p5 = features
        
        # FPN (top-down)
        fpn_p5 = p5
        fpn_p4 = self.fpn_p4(torch.cat([self.upsample(fpn_p5), p4], 1))
        fpn_p3 = self.fpn_p3(torch.cat([self.upsample(fpn_p4), p3], 1))
        
        # PAN (bottom-up)
        pan_p3 = fpn_p3
        pan_p4 = self.pan_p4(torch.cat([self.downsample_p3(pan_p3), fpn_p4], 1))
        pan_p5 = self.pan_p5(torch.cat([self.downsample_p4(pan_p4), fpn_p5], 1))
        
        return pan_p3, pan_p4, pan_p5