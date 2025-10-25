import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import BackboneComponents as bc
import RadarComponents as rc
import FusionComponents as fc
import DetectionHead as dh

class ProbabilisticYOLORadar3D(nn.Module):
    """Complete Probabilistic YOLO-Radar 3D Detection Model"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Image backbone
        self.image_backbone = bc.YOLOv8Backbone()
        
        # Radar branch
        self.radar_branch = rc.ProbabilisticRadarBranch()
        
        # Radar preprocessor
        self.radar_processor = rc.RadarGaussianProcessor()
        
        # Fusion modules
        self.fusion_p3 = fc.UncertaintyAwareFusion(channels=256)
        self.fusion_p4 = fc.UncertaintyAwareFusion(channels=512)
        self.fusion_p5 = fc.UncertaintyAwareFusion(channels=1024)
        
        # Neck
        self.neck = bc.YOLOv8Neck()
        
        # Detection head
        self.head_3d = dh.ProbabilisticDetection3DHead(
            num_classes=num_classes,
            in_channels=[256, 512, 1024]
        )
    
    def forward(self, image, radar_points, radar_timestamps, calib):
        batch_size = image.size(0)
        
        # Use current timestamp (placeholder)
        image_timestamp = torch.zeros(batch_size, device=image.device)
        
        # Process radar
        gaussian_spheres = self.radar_processor(
            radar_points, radar_timestamps, image_timestamp
        )
        
        # Extract image features
        img_p3, img_p4, img_p5 = self.image_backbone(image)
        
        # Process radar
        radar_p3, radar_p4, radar_p5, uncertainty_maps = \
            self.radar_branch(gaussian_spheres, calib)
        
        # Fusion
        fused_p3, alpha_p3 = self.fusion_p3(img_p3, radar_p3, uncertainty_maps['p3'])
        fused_p4, alpha_p4 = self.fusion_p4(img_p4, radar_p4, uncertainty_maps['p4'])
        fused_p5, alpha_p5 = self.fusion_p5(img_p5, radar_p5, uncertainty_maps['p5'])
        
        # Neck
        neck_features = self.neck([fused_p3, fused_p4, fused_p5])
        
        # Detection (use P3 for simplicity)
        predictions = self.head_3d(neck_features[0], uncertainty_maps['p3'])
        
        # Add fusion weights
        predictions['fusion_weights'] = {
            'p3': alpha_p3,
            'p4': alpha_p4,
            'p5': alpha_p5
        }
        
        return predictions