import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import BackboneComponents as bc

class UncertaintyAwareFusion(nn.Module):
    """Fuse image and radar features with uncertainty weighting"""
    def __init__(self, channels):
        super().__init__()
        self.conv_img = bc.Conv(channels, channels, 1)
        self.conv_radar = bc.Conv(channels, channels, 1)
        self.fusion_weight = nn.Sequential(
            bc.Conv(channels * 2 + 1, 64, 1),
            nn.ReLU(),
            bc.Conv(64, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, img_feat, radar_feat, uncertainty_map):
        img_proj = self.conv_img(img_feat)
        radar_proj = self.conv_radar(radar_feat)
        
        # Resize uncertainty map if needed
        if uncertainty_map.shape[-2:] != img_proj.shape[-2:]:
            uncertainty_map = F.interpolate(
                uncertainty_map, size=img_proj.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Compute fusion weights
        concat = torch.cat([img_proj, radar_proj, uncertainty_map], dim=1)
        weights = self.fusion_weight(concat)
        
        alpha_img = weights[:, 0:1]
        alpha_radar = weights[:, 1:2]
        
        # Fuse
        fused = alpha_img * img_proj + alpha_radar * radar_proj
        fused = fused + img_feat  # Residual
        
        return fused, alpha_radar