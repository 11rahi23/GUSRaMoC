import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import BackboneComponents as bc

class RadarGaussianProcessor(nn.Module):
    """Process radar points into Gaussian spheres"""
    def __init__(self):
        super().__init__()
        self.range_accuracy = 0.1
        self.azimuth_accuracy = 2.0 * np.pi / 180
        self.elevation_accuracy = 5.0 * np.pi / 180
        self.doppler_accuracy = 0.1
        self.register_buffer('dummy', torch.zeros(1))  # For device handling
    
    def forward(self, radar_points, timestamps, img_timestamp):
        """
        Args:
            radar_points: (B, N, 6) [x, y, z, vx, vy, RCS]
            timestamps: (B, N) timestamps
            img_timestamp: (B,) image timestamp
        Returns:
            gaussian_data: (B, N, 14)
        """
        B, N, _ = radar_points.shape
        device = radar_points.device
        
        positions = radar_points[..., :3]
        velocities = radar_points[..., 3:5]
        rcs = radar_points[..., 5:6]
        
        # Time differences
        dt = img_timestamp.unsqueeze(1) - timestamps
        
        # Compute uncertainties
        ranges = torch.norm(positions, dim=-1, keepdim=True)
        
        # Simplified Cartesian uncertainties
        sigma_x = torch.full_like(ranges, self.range_accuracy)
        sigma_y = torch.full_like(ranges, self.range_accuracy)
        sigma_z = torch.full_like(ranges, self.range_accuracy)
        
        # Add temporal uncertainty
        temporal_unc = self.doppler_accuracy * dt.unsqueeze(-1)
        sigma_x = torch.sqrt(sigma_x**2 + temporal_unc**2)
        sigma_y = torch.sqrt(sigma_y**2 + temporal_unc**2)
        
        # Velocity uncertainties
        sigma_vx = torch.full_like(velocities[..., 0:1], self.doppler_accuracy)
        sigma_vy = torch.full_like(velocities[..., 1:2], self.doppler_accuracy)
        
        # RCS-based weight
        rcs_normalized = (rcs - rcs.min()) / (rcs.max() - rcs.min() + 1e-6)
        weight = torch.sigmoid(5 * (rcs_normalized - 0.5))
        
        # Azimuth
        azimuths = torch.atan2(positions[..., 1:2], positions[..., 0:1])
        
        # Combine
        gaussian_data = torch.cat([
            positions, velocities, sigma_x, sigma_y, sigma_z,
            sigma_vx, sigma_vy, dt.unsqueeze(-1), rcs, weight, azimuths
        ], dim=-1)
        
        return gaussian_data


class ProbabilisticRadarEncoder(nn.Module):
    """Encode radar Gaussian spheres to features"""
    def __init__(self, input_dim=14, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, gaussian_data):
        # gaussian_data: (B, N, 14)
        return self.mlp(gaussian_data)  # (B, N, 256)


class ProbabilisticRadarBranch(nn.Module):
    """Probabilistic radar processing branch"""
    def __init__(self):
        super().__init__()
        self.encoder = ProbabilisticRadarEncoder(input_dim=14, output_dim=256)
        
        # Simple CNN for feature maps (placeholder)
        self.conv_p3 = nn.Sequential(
            bc.Conv(256, 256, 3, 1),
            bc.Conv(256, 256, 3, 1)
        )
        self.conv_p4 = nn.Sequential(
            bc.Conv(256, 512, 3, 2),
            bc.Conv(512, 512, 3, 1)
        )
        self.conv_p5 = nn.Sequential(
            bc.Conv(512, 1024, 3, 2),
            bc.Conv(1024, 1024, 3, 1)
        )
    
    def forward(self, gaussian_data, calib):
        B, N, _ = gaussian_data.shape
        
        # Encode features
        point_features = self.encoder(gaussian_data)  # (B, N, 256)
        
        # Create pseudo feature maps (simplified projection)
        # In practice, project to image space properly
        H, W = 80, 80
        pseudo_map = point_features.mean(dim=1).view(B, 256, 1, 1).expand(B, 256, H, W)
        
        # Multi-scale features
        p3 = self.conv_p3(pseudo_map)
        p4 = self.conv_p4(p3)
        p5 = self.conv_p5(p4)
        
        # Uncertainty maps (simplified)
        uncertainty_maps = {
            'p3': torch.ones(B, 1, H, W, device=gaussian_data.device) * 0.1,
            'p4': torch.ones(B, 1, H//2, W//2, device=gaussian_data.device) * 0.1,
            'p5': torch.ones(B, 1, H//4, W//4, device=gaussian_data.device) * 0.1
        }
        
        return p3, p4, p5, uncertainty_maps