import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix

class NuScenesRadarDataset(Dataset):
    """
    nuScenes dataset loader for camera + radar fusion
    """
    def __init__(self, 
                 dataroot='/data/sets/nuscenes',
                 version='v1.0-trainval',
                 split='train',
                 image_size=(640, 640),
                 max_radar_points=1000):
        """
        Args:
            dataroot: Path to nuScenes dataset
            version: 'v1.0-trainval' or 'v1.0-mini'
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            max_radar_points: Maximum number of radar points to use
        """
        self.dataroot = dataroot
        self.image_size = image_size
        self.max_radar_points = max_radar_points
        
        # Initialize nuScenes
        print(f"Loading nuScenes {version} - {split} split...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        # Get samples for the split
        self.samples = self._get_samples(split)
        print(f"Found {len(self.samples)} samples")
        
        # Camera sensor
        self.camera_channel = 'CAM_FRONT'
        
        # Radar sensors (use all 5 radars, but focus on front)
        self.radar_channel = 'RADAR_FRONT'
        
        # nuScenes classes (10 detection classes)
        self.class_names = [
            'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
            'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
        ]
        
    def _get_samples(self, split):
        """Get samples for train/val/test split"""
        # nuScenes official split
        if split == 'train':
            split_logs = [
                'scene-0001', 'scene-0002', 'scene-0003', 'scene-0004',
                'scene-0005', 'scene-0006', 'scene-0007', 'scene-0008',
                'scene-0009', 'scene-0010'
            ]  # Simplified - use all training scenes
        elif split == 'val':
            split_logs = ['scene-0011', 'scene-0012']  # Simplified
        else:
            raise ValueError(f"Unknown split: {split}")
        
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            if split == 'train' or split == 'val':  # Include all for now
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) tensor
            radar_points: (N, 6) tensor [x, y, z, vx, vy, RCS]
            radar_timestamps: (N,) tensor
            calibration: dict with intrinsic/extrinsic matrices
            targets: dict with 3D bounding boxes and labels
        """
        sample = self.samples[idx]
        
        # 1. Load camera image
        cam_data = self.nusc.get('sample_data', sample['data'][self.camera_channel])
        image_path = os.path.join(self.dataroot, cam_data['filename'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get camera calibration
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
        
        # Get camera pose
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        cam_extrinsic = transform_matrix(
            cam_calib['translation'],
            Quaternion(cam_calib['rotation']),
            inverse=False
        )
        
        # 2. Load radar point clouds from all radars
        radar_points_list = []
        radar_timestamps_list = []
        image_timestamp = cam_data['timestamp']
        
        
        radar_data = self.nusc.get('sample_data', sample['data'][self.radar_channel])
        radar_path = os.path.join(self.dataroot, radar_data['filename'])
            
        # Load radar points
        radar_pc = RadarPointCloud.from_file(radar_path)
            
        # Get radar calibration
        radar_calib = self.nusc.get('calibrated_sensor', 
                                       radar_data['calibrated_sensor_token'])
            
        # Transform radar points to camera frame
        points = radar_pc.points[:3, :]  # x, y, z
        velocities = radar_pc.points[8:10, :]  # vx, vy (compensated)
        rcs = radar_pc.points[5:6, :]  # RCS
            
        # Transform from radar frame to ego vehicle frame
        radar_to_ego = transform_matrix(
            radar_calib['translation'],
            Quaternion(radar_calib['rotation']),
            inverse=False
        )
            
        # Transform from ego to camera frame
        ego_to_cam = np.linalg.inv(cam_extrinsic)
        radar_to_cam = ego_to_cam @ radar_to_ego
            
        # Apply transformation
        points_homo = np.vstack([points, np.ones((1, points.shape[1]))])
        points_cam = (radar_to_cam @ points_homo)[:3, :]
            
        # Filter points in front of camera
        valid_mask = points_cam[2, :] > 0
            
        if valid_mask.sum() > 0:
            points_cam = points_cam[:, valid_mask]
            velocities_cam = velocities[:, valid_mask]
            rcs_cam = rcs[:, valid_mask]
                
            # Combine into (N, 6) array
            radar_points = np.vstack([
                points_cam,
                velocities_cam,
                rcs_cam
            ]).T  # (N, 6)
                
            radar_points_list.append(radar_points)
                
            # Timestamps (time difference from image)
            dt = (image_timestamp - radar_data['timestamp']) / 1e6  # Convert to seconds
            radar_timestamps_list.append(
                np.full(radar_points.shape[0], dt)
            )
        
        # Concatenate all radar points
        if len(radar_points_list) > 0:
            all_radar_points = np.vstack(radar_points_list)
            all_radar_timestamps = np.concatenate(radar_timestamps_list)
        else:
            # No radar points available
            all_radar_points = np.zeros((0, 6))
            all_radar_timestamps = np.zeros((0,))
        
        # Subsample if too many points
        if all_radar_points.shape[0] > self.max_radar_points:
            indices = np.random.choice(
                all_radar_points.shape[0],
                self.max_radar_points,
                replace=False
            )
            all_radar_points = all_radar_points[indices]
            all_radar_timestamps = all_radar_timestamps[indices]
        
        # Pad if too few points
        if all_radar_points.shape[0] < self.max_radar_points:
            padding = self.max_radar_points - all_radar_points.shape[0]
            all_radar_points = np.vstack([
                all_radar_points,
                np.zeros((padding, 6))
            ])
            all_radar_timestamps = np.concatenate([
                all_radar_timestamps,
                np.zeros(padding)
            ])
        
        # 3. Get 3D annotations
        targets = self._get_annotations(sample, cam_intrinsic, cam_extrinsic)
        
        # 4. Preprocess image
        image, scale_factor = self._preprocess_image(image)
        
        # Update calibration with scale factor
        cam_intrinsic_scaled = cam_intrinsic.copy()
        cam_intrinsic_scaled[0, :] *= scale_factor[1]  # width
        cam_intrinsic_scaled[1, :] *= scale_factor[0]  # height
        
        # 5. Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        radar_points_tensor = torch.from_numpy(all_radar_points).float()
        radar_timestamps_tensor = torch.from_numpy(all_radar_timestamps).float()
        
        calibration = {
            'intrinsic': torch.from_numpy(cam_intrinsic_scaled).float(),
            'extrinsic': torch.from_numpy(cam_extrinsic).float()
        }
        
        return {
            'image': image_tensor,
            'radar_points': radar_points_tensor,
            'radar_timestamps': radar_timestamps_tensor,
            'calibration': calibration,
            'targets': targets,
            'sample_token': sample['token']
        }
    
    def _preprocess_image(self, image):
        """Resize image to target size"""
        h, w = image.shape[:2]
        target_h, target_w = self.image_size
        
        # Resize
        image_resized = cv2.resize(image, (target_w, target_h))
        
        scale_factor = (target_h / h, target_w / w)
        
        return image_resized, scale_factor
    
    def _get_annotations(self, sample, cam_intrinsic, cam_extrinsic):
        """Get 3D bounding box annotations in camera frame"""
        boxes_3d = []
        labels = []
        velocities = []
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Filter by class
            if ann['category_name'].split('.')[0] not in ['vehicle', 'human', 'movable_object']:
                continue
            
            # Get box in global frame
            box = self.nusc.get_box(ann_token)
            
            # Transform to camera frame
            box.translate(-np.array(cam_extrinsic[:3, 3]))
            box.rotate(Quaternion(matrix=cam_extrinsic[:3, :3]).inverse)
            
            # Check if box is in front of camera
            if box.center[2] <= 0:
                continue
            
            # Project box center to image
            center_3d = box.center
            center_2d = view_points(
                center_3d.reshape(3, 1),
                cam_intrinsic,
                normalize=True
            ).T[0, :2]
            
            # Get class label
            category = ann['category_name']
            if 'vehicle.car' in category:
                label = 0
            elif 'vehicle.truck' in category:
                label = 1
            elif 'vehicle.bus' in category:
                label = 2
            elif 'vehicle.construction' in category:
                label = 3
            elif 'vehicle.trailer' in category:
                label = 4
            elif 'human.pedestrian' in category:
                label = 5
            elif 'vehicle.motorcycle' in category:
                label = 6
            elif 'vehicle.bicycle' in category:
                label = 7
            elif 'movable_object.trafficcone' in category:
                label = 8
            elif 'movable_object.barrier' in category:
                label = 9
            elif 'movable_object.pushable_pullable' in category:
                label = 9
            else:
                continue  # Skip other classes for simplicity
            
            # Get velocity
            velocity = self.nusc.box_velocity(ann_token)[:2]  # vx, vy in global frame
            
            # Store annotation
            boxes_3d.append({
                'center_3d': center_3d,
                'dimensions': box.wlh,  # width, length, height
                'rotation': box.orientation.yaw_pitch_roll[0],  # yaw angle
                'center_2d': center_2d
            })
            labels.append(label)
            velocities.append(velocity)
        
        return {
            'boxes_3d': boxes_3d,
            'labels': labels,
            'velocities': velocities,
            'num_boxes': len(boxes_3d)
        }
