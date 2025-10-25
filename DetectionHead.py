import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import BackboneComponents as bc

class ProbabilisticDetection3DHead(nn.Module):
    """3D detection head with uncertainty estimation"""
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.num_classes = num_classes
        
        # For simplicity, use same head for all scales
        # In practice, create separate heads for each scale
        c = in_channels[0]  # Use first channel count
        
        # Classification
        self.cls_convs = nn.Sequential(
            bc.Conv(c, c, 3, 1),
            bc.Conv(c, c, 3, 1)
        )
        self.cls_pred = nn.Conv2d(c, num_classes, 1)
        
        # 2D box
        self.box_2d_convs = nn.Sequential(
            bc.Conv(c, c, 3, 1),
            bc.Conv(c, c, 3, 1)
        )
        self.box_2d_pred = nn.Conv2d(c, 4, 1)
        
        # 3D attributes
        self.box_3d_convs = nn.Sequential(
            bc.Conv(c, c, 3, 1),
            bc.Conv(c, c, 3, 1)
        )
        self.depth_mean = nn.Conv2d(c, 1, 1)
        self.depth_std = nn.Conv2d(c, 1, 1)
        self.center_offset = nn.Conv2d(c, 2, 1)
        self.dims_mean = nn.Conv2d(c, 3, 1)
        self.rot_mean = nn.Conv2d(c, 2, 1)
        self.vel_mean = nn.Conv2d(c, 2, 1)
        self.vel_std = nn.Conv2d(c, 2, 1)
    
    def forward(self, x, radar_uncertainties=None):
        # Classification
        cls_feat = self.cls_convs(x)
        cls_out = self.cls_pred(cls_feat)
        
        # 2D boxes
        box_2d_feat = self.box_2d_convs(x)
        box_2d_out = self.box_2d_pred(box_2d_feat)
        
        # 3D attributes
        box_3d_feat = self.box_3d_convs(x)
        depth_mean = self.depth_mean(box_3d_feat)
        depth_std = F.softplus(self.depth_std(box_3d_feat)) + 1e-6
        center_offset = self.center_offset(box_3d_feat)
        dims_mean = self.dims_mean(box_3d_feat)
        rot_mean = self.rot_mean(box_3d_feat)
        vel_mean = self.vel_mean(box_3d_feat)
        vel_std = F.softplus(self.vel_std(box_3d_feat)) + 1e-6
        
        return {
            'cls': cls_out,
            'box_2d': box_2d_out,
            'depth_mean': torch.exp(depth_mean),
            'depth_std': depth_std,
            'center_offset': center_offset,
            'dims_mean': torch.exp(dims_mean),
            'rot_mean': rot_mean,
            'vel_mean': vel_mean,
            'vel_std': vel_std
        }