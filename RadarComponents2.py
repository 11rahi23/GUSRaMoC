"""
Probabilistic Radar Branch with Gaussian Projection and Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianRadarProjection(nn.Module):
    """
    Project radar Gaussian spheres onto image plane
    Uses the uncertainty information to weight contributions
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Learnable projection for combining spatial and feature information
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, point_features, gaussian_data, calib, image_size):
        """
        Args:
            point_features: (B, N, feature_dim) - encoded radar features
            gaussian_data: (B, N, 14) - Gaussian sphere data
            calib: List of calibration dicts (length B)
            image_size: (H, W) - target feature map size
            
        Returns:
            feature_map: (B, feature_dim, H, W)
            uncertainty_map: (B, 1, H, W)
            density_map: (B, 1, H, W)
        """
        B, N, C = point_features.shape
        H, W = image_size
        device = point_features.device
        
        # Extract Gaussian parameters
        positions = gaussian_data[..., :3]  # (B, N, 3) [x, y, z]
        sigma_x = gaussian_data[..., 5:6]   # (B, N, 1)
        sigma_y = gaussian_data[..., 6:7]   # (B, N, 1)
        weights = gaussian_data[..., 12:13] # (B, N, 1) RCS-based weight
        
        # Initialize output maps
        feature_map = torch.zeros(B, C, H, W, device=device)
        uncertainty_map = torch.zeros(B, 1, H, W, device=device)
        density_map = torch.zeros(B, 1, H, W, device=device)
        
        # Transform features
        point_features_transformed = self.feature_transform(point_features)
        
        # Process each sample in batch
        for b in range(B):
            if calib[b] is None:
                continue
                
            # Get calibration matrices
            intrinsic = calib[b]['intrinsic']  # (3, 3)
            if isinstance(intrinsic, torch.Tensor):
                intrinsic = intrinsic.cpu().numpy()

            # Project 3D points to 2D image
            points_3d = positions[b].cpu().numpy()  # (N, 3)
            
            # Filter points in front of camera
            valid_mask = points_3d[:, 2] > 0.1
            if valid_mask.sum() == 0:
                continue
            
            points_3d_valid = points_3d[valid_mask]
            
            # Project to image coordinates
            points_2d_homo = intrinsic @ points_3d_valid.T  # (3, N_valid)
            points_2d = points_2d_homo[:2] / (points_2d_homo[2:3] + 1e-6)  # (2, N_valid)
            points_2d = points_2d.T  # (N_valid, 2)
            
            # Scale to feature map size
            scale_x = W / intrinsic[0, 2] / 2
            scale_y = H / intrinsic[1, 2] / 2
            points_2d[:, 0] *= scale_x
            points_2d[:, 1] *= scale_y
            
            # Get valid features and parameters
            valid_features = point_features_transformed[b, valid_mask].to(device)  # (N_valid, C)
            valid_sigma_x = sigma_x[b, valid_mask].to(device)  # (N_valid, 1)
            valid_sigma_y = sigma_y[b, valid_mask].to(device)  # (N_valid, 1)
            valid_weights = weights[b, valid_mask].to(device)  # (N_valid, 1)
            
            # Create coordinate grid
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
            
            # Gaussian splatting for each point
            for i in range(len(points_2d)):
                center = torch.from_numpy(points_2d[i]).to(device)  # (2,)
                
                # Skip if outside image
                if center[0] < 0 or center[0] >= W or center[1] < 0 or center[1] >= H:
                    continue
                
                # Compute distance from each pixel to point center
                diff = grid - center.view(1, 1, 2)  # (H, W, 2)
                
                # Mahalanobis distance with uncertainty
                sigma_x_val = valid_sigma_x[i].item() * scale_x
                sigma_y_val = valid_sigma_y[i].item() * scale_y
                
                # Prevent too small sigma (Is it necessary? As it is measuring uncertainty)
                sigma_x_val = max(sigma_x_val, 1.0)
                sigma_y_val = max(sigma_y_val, 1.0)
                
                # Gaussian weight computation
                dist_x = (diff[..., 0] / sigma_x_val) ** 2
                dist_y = (diff[..., 1] / sigma_y_val) ** 2
                gaussian_weight = torch.exp(-0.5 * (dist_x + dist_y))  # (H, W)
                
                # Apply RCS-based confidence weight
                gaussian_weight = gaussian_weight * valid_weights[i, 0]
                
                # Only consider pixels within 3 sigma
                mask = (dist_x + dist_y) < 9.0  # 3 sigma threshold
                gaussian_weight = gaussian_weight * mask.float()
                
                # Accumulate features
                feature_contribution = valid_features[i:i+1].view(C, 1, 1) * gaussian_weight.unsqueeze(0)
                feature_map[b] += feature_contribution
                
                # Accumulate uncertainty
                avg_sigma = (sigma_x_val + sigma_y_val) / 2
                uncertainty_contribution = gaussian_weight * avg_sigma
                uncertainty_map[b, 0] += uncertainty_contribution
                
                # Accumulate density
                density_map[b, 0] += gaussian_weight
        
        # Normalize by density
        density_map = density_map.clamp(min=1e-6)
        feature_map = feature_map / density_map
        uncertainty_map = uncertainty_map / density_map
        
        return feature_map, uncertainty_map, density_map


class CrossModalAttention(nn.Module):
    """
    Cross-attention between image and radar features
    Allows image features to attend to radar features and vice versa
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Separate Q, K, V projections for image and radar
        self.q_img = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_radar = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_radar = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_img = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_img = nn.Linear(dim, dim)
        self.proj_radar = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, img_features, radar_features, uncertainty_mask=None):
        """
        Args:
            img_features: (B, C, H, W)
            radar_features: (B, C, H, W)
            uncertainty_mask: (B, 1, H, W) - optional uncertainty weighting
            
        Returns:
            img_attended: (B, C, H, W)
            radar_attended: (B, C, H, W)
        """
        B, C, H, W = img_features.shape
        
        # Reshape to sequence format
        img_seq = img_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        radar_seq = radar_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        N = H * W
        
        # ===== Image attends to Radar =====
        q_img = self.q_img(img_seq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_radar = self.kv_radar(radar_seq).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_radar, v_radar = kv_radar[0], kv_radar[1]
        
        attn_img2radar = (q_img @ k_radar.transpose(-2, -1)) * self.scale
        
        # Apply uncertainty mask if provided
        if uncertainty_mask is not None:
            uncertainty_seq = uncertainty_mask.flatten(2).transpose(1, 2)  # (B, H*W, 1)
            # Lower uncertainty = higher attention weight
            uncertainty_weight = 1.0 / (uncertainty_seq + 1e-6)
            uncertainty_weight = uncertainty_weight.unsqueeze(1)  # (B, 1, H*W, 1)
            attn_img2radar = attn_img2radar * uncertainty_weight
        
        attn_img2radar = attn_img2radar.softmax(dim=-1)
        attn_img2radar = self.attn_drop(attn_img2radar)
        
        img_attended_seq = (attn_img2radar @ v_radar).transpose(1, 2).reshape(B, N, C)
        img_attended_seq = self.proj_img(img_attended_seq)
        img_attended_seq = self.proj_drop(img_attended_seq)
        
        # ===== Radar attends to Image =====
        q_radar = self.q_radar(radar_seq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_img = self.kv_img(img_seq).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_img, v_img = kv_img[0], kv_img[1]
        
        attn_radar2img = (q_radar @ k_img.transpose(-2, -1)) * self.scale
        attn_radar2img = attn_radar2img.softmax(dim=-1)
        attn_radar2img = self.attn_drop(attn_radar2img)
        
        radar_attended_seq = (attn_radar2img @ v_img).transpose(1, 2).reshape(B, N, C)
        radar_attended_seq = self.proj_radar(radar_attended_seq)
        radar_attended_seq = self.proj_drop(radar_attended_seq)
        
        # Reshape back to spatial format
        img_attended = img_attended_seq.transpose(1, 2).reshape(B, C, H, W)
        radar_attended = radar_attended_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return img_attended, radar_attended


class RadarImageFusionBlock(nn.Module):
    """
    Fusion block combining radar and image features with cross-attention
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        
        # Layer normalization
        self.norm_img = nn.GroupNorm(32, channels)
        self.norm_radar = nn.GroupNorm(32, channels)
        
        # Cross-attention
        self.cross_attn = CrossModalAttention(
            dim=channels,
            num_heads=num_heads,
            attn_drop=0.1,
            proj_drop=0.1
        )
        
        # Feed-forward networks
        self.ffn_img = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Dropout(0.1)
        )
        
        self.ffn_radar = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Dropout(0.1)
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        
    def forward(self, img_feat, radar_feat, uncertainty_map=None):
        """
        Args:
            img_feat: (B, C, H, W)
            radar_feat: (B, C, H, W)
            uncertainty_map: (B, 1, H, W)
        """
        # Normalize
        img_norm = self.norm_img(img_feat)
        radar_norm = self.norm_radar(radar_feat)
        
        # Cross-attention
        img_attn, radar_attn = self.cross_attn(img_norm, radar_norm, uncertainty_map)
        
        # Residual connection
        img_attn = img_feat + img_attn
        radar_attn = radar_feat + radar_attn
        
        # Feed-forward
        img_out = img_attn + self.ffn_img(img_attn)
        radar_out = radar_attn + self.ffn_radar(radar_attn)
        
        # Fuse
        fused = torch.cat([img_out, radar_out], dim=1)
        fused = self.fusion_conv(fused)
        
        return fused


class ImprovedProbabilisticRadarBranch(nn.Module):
    """
    Improved Probabilistic Radar Branch with:
    1. Gaussian sphere projection to image space
    2. Cross-attention with image features
    3. Multi-scale processing
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Radar encoder
        self.encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, feature_dim)
        )
        
        # Gaussian projection module
        self.gaussian_projection = GaussianRadarProjection(feature_dim=feature_dim)
        
        # Multi-scale fusion with cross-attention
        self.fusion_p3 = RadarImageFusionBlock(channels=256, num_heads=8)
        self.fusion_p4 = RadarImageFusionBlock(channels=512, num_heads=8)
        self.fusion_p5 = RadarImageFusionBlock(channels=1024, num_heads=8)
        
        # Downsample radar features to match different scales
        self.downsample_p4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU()
        )
        
        self.downsample_p5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU()
        )
        
    def forward(self, gaussian_data, calib, img_features=None):
        """
        Args:
            gaussian_data: (B, N, 14) - Gaussian sphere representation
            calib: List of calibration dicts
            img_features: Dict or tuple with image features at different scales
                         {'p3': (B, 256, H, W), 'p4': (B, 512, H/2, W/2), 'p5': (B, 1024, H/4, W/4)}
                         or (p3, p4, p5) tuple
                         
        Returns:
            radar_p3, radar_p4, radar_p5: Multi-scale radar features
            uncertainty_maps: Dict of uncertainty maps
            fused_features: Dict of fused features (if img_features provided)
        """
        B, N, _ = gaussian_data.shape
        device = gaussian_data.device
        
        # Encode radar points
        point_features = self.encoder(gaussian_data)  # (B, N, 256)
        
        # Determine image size from img_features if provided
        if img_features is not None:
            if isinstance(img_features, dict):
                img_p3 = img_features['p3']
            elif isinstance(img_features, (tuple, list)):
                img_p3 = img_features[0]
            else:
                img_p3 = img_features
            _, _, H, W = img_p3.shape
        else:
            H, W = 80, 80  # Default
        
        # Project to image space using Gaussian splatting
        radar_p3, uncertainty_p3, density_p3 = self.gaussian_projection(
            point_features, gaussian_data, calib, image_size=(H, W)
        )
        
        # Generate multi-scale features
        radar_p4 = self.downsample_p4(radar_p3)
        radar_p5 = self.downsample_p5(radar_p4)
        
        # Downsample uncertainty maps
        uncertainty_p4 = F.interpolate(uncertainty_p3, scale_factor=0.5, mode='bilinear', align_corners=False)
        uncertainty_p5 = F.interpolate(uncertainty_p4, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        uncertainty_maps = {
            'p3': uncertainty_p3,
            'p4': uncertainty_p4,
            'p5': uncertainty_p5,
            'density_p3': density_p3
        }
        
        # If image features provided, perform cross-modal fusion
        fused_features = None
        if img_features is not None:
            if isinstance(img_features, dict):
                img_p3, img_p4, img_p5 = img_features['p3'], img_features['p4'], img_features['p5']
            else:
                img_p3, img_p4, img_p5 = img_features
            
            fused_p3 = self.fusion_p3(img_p3, radar_p3, uncertainty_p3)
            fused_p4 = self.fusion_p4(img_p4, radar_p4, uncertainty_p4)
            fused_p5 = self.fusion_p5(img_p5, radar_p5, uncertainty_p5)
            
            fused_features = {
                'p3': fused_p3,
                'p4': fused_p4,
                'p5': fused_p5
            }
        
        return radar_p3, radar_p4, radar_p5, uncertainty_maps, fused_features


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    print("Testing Improved Probabilistic Radar Branch...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create module
    radar_branch = ImprovedProbabilisticRadarBranch(feature_dim=256).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in radar_branch.parameters())
    print(f"Radar branch parameters: {num_params:,}")
    
    # Create dummy data
    B = 2
    N = 1000
    H, W = 80, 80
    
    # Gaussian data: [x, y, z, vx, vy, σx, σy, σz, σvx, σvy, dt, rcs, weight, azimuth]
    gaussian_data = torch.randn(B, N, 14).to(device)
    gaussian_data[..., :3] = torch.randn(B, N, 3).to(device) * 10  # positions
    gaussian_data[..., 5:8] = torch.rand(B, N, 3).to(device) * 0.5  # uncertainties
    gaussian_data[..., 12] = torch.rand(B, N, 1).to(device)  # weights
    
    # Calibration (simplified)
    intrinsic = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
    intrinsic[:, 0, 0] = 1000  # fx
    intrinsic[:, 1, 1] = 1000  # fy
    intrinsic[:, 0, 2] = W / 2  # cx
    intrinsic[:, 1, 2] = H / 2  # cy
    
    calib = [
        {'intrinsic': intrinsic[i], 'extrinsic': torch.eye(4)}
        for i in range(B)
    ]
    
    # Image features
    img_p3 = torch.randn(B, 256, H, W).to(device)
    img_p4 = torch.randn(B, 512, H//2, W//2).to(device)
    img_p5 = torch.randn(B, 1024, H//4, W//4).to(device)
    img_features = (img_p3, img_p4, img_p5)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        radar_p3, radar_p4, radar_p5, uncertainty_maps, fused_features = radar_branch(
            gaussian_data, calib, img_features
        )
    
    print("\nOutput shapes:")
    print(f"  Radar P3: {radar_p3.shape}")
    print(f"  Radar P4: {radar_p4.shape}")
    print(f"  Radar P5: {radar_p5.shape}")
    print(f"  Uncertainty P3: {uncertainty_maps['p3'].shape}")
    print(f"  Fused P3: {fused_features['p3'].shape}")
    
    print("\n✓ Test passed!")