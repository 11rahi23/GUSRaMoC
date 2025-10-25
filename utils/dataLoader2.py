import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from sklearn.cluster import DBSCAN

class NuScenesRadarDataset(Dataset):
    """
    nuScenes dataset loader for camera + radar fusion with grid-based clustering
    """
    def __init__(self, 
                 dataroot='/data/sets/nuscenes',
                 version='v1.0-trainval',
                 split='train',
                 image_size=(640, 640),
                 max_radar_points=1000,
                 time_threshold=500000,
                 grid_size=1.0,  # NEW: Grid size in meters
                 dbscan_eps=1.0/np.sqrt(3),  # NEW: DBSCAN epsilon
                 dbscan_min_pts=3):  # NEW: DBSCAN min points
        """
        Args:
            dataroot: Path to nuScenes dataset
            version: 'v1.0-trainval' or 'v1.0-mini'
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            max_radar_points: Maximum number of radar points to use
            time_threshold: Time threshold for radar sweeps in microseconds
            grid_size: Grid size in meters (default: 1m)
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_pts: DBSCAN minimum points parameter
        """
        self.dataroot = dataroot
        self.image_size = image_size
        self.max_radar_points = max_radar_points
        self.time_threshold = time_threshold
        self.grid_size = grid_size
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_pts = dbscan_min_pts
        
        # Initialize nuScenes
        print(f"Loading nuScenes {version} - {split} split...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        # Get samples for the split
        self.samples = self._get_samples(split)
        print(f"Found {len(self.samples)} samples")
        
        # Camera sensor
        self.camera_channel = 'CAM_FRONT'
        
        # Radar sensors
        self.radar_channel = 'RADAR_FRONT'
        
        # nuScenes classes (10 detection classes)
        self.class_names = [
            'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
            'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
        ]
        
    def _get_samples(self, split):
        """Get samples for train/val/test split"""
        if split == 'train':
            split_logs = [
                'scene-0001', 'scene-0002', 'scene-0003', 'scene-0004',
                'scene-0005', 'scene-0006', 'scene-0007', 'scene-0008',
                'scene-0009', 'scene-0010'
            ]
        elif split == 'val':
            split_logs = ['scene-0011', 'scene-0012']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            if split == 'train' or split == 'val':
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _find_radar_sweeps(self, cam_timestamp, radar_channel):
        """Find all radar sweeps within time threshold of camera timestamp"""
        radar_sweeps = []
        
        for sample in self.nusc.sample:
            if radar_channel not in sample['data']:
                continue
            
            radar_token = sample['data'][radar_channel]
            radar_sd = self.nusc.get('sample_data', radar_token)
            
            current_radar = radar_sd
            while current_radar is not None:
                time_diff = abs(current_radar['timestamp'] - cam_timestamp)
                
                if time_diff < self.time_threshold:
                    radar_sweeps.append(current_radar)
                
                if current_radar['next'] != '':
                    current_radar = self.nusc.get('sample_data', current_radar['next'])
                else:
                    break
        
        return radar_sweeps
    
    def _assign_to_grid(self, points):
        """
        Assign 3D points to grid cells
        
        Args:
            points: (N, 3) array of xyz coordinates
            
        Returns:
            grid_indices: (N, 3) array of grid cell indices
        """
        grid_indices = np.floor(points / self.grid_size).astype(np.int32)
        return grid_indices
    
    def _grid_dbscan_clustering(self, points, timestamps, velocities, rcs):
        """
        Apply grid-based DBSCAN clustering
        
        Args:
            points: (N, 3) array of xyz coordinates in camera frame
            timestamps: (N,) array of time differences
            velocities: (N, 2) array of vx, vy
            rcs: (N,) array of RCS values
            
        Returns:
            clustered_points: (M, 10) array [x, y, z, vx, vy, confidence, std_x, std_y, std_z, rcs]
            where M is the number of clusters
        """
        if points.shape[0] == 0:
            return np.zeros((0, 10))
        
        # Step 1: Assign points to grid cells
        grid_indices = self._assign_to_grid(points)
        
        # Create unique grid cell identifiers
        grid_ids = {}
        for i in range(points.shape[0]):
            grid_key = tuple(grid_indices[i])
            if grid_key not in grid_ids:
                grid_ids[grid_key] = []
            grid_ids[grid_key].append(i)
        
        # Step 2: Apply DBSCAN within each non-empty grid and its neighbors
        all_labels = -np.ones(points.shape[0], dtype=np.int32)
        cluster_id = 0
        
        processed_grids = set()
        
        for grid_key in grid_ids.keys():
            if grid_key in processed_grids:
                continue
            
            # Get points in this grid and neighboring grids (26-connectivity)
            neighbor_points = []
            neighbor_indices = []
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_key = (grid_key[0] + dx, grid_key[1] + dy, grid_key[2] + dz)
                        if neighbor_key in grid_ids:
                            neighbor_points.extend([points[i] for i in grid_ids[neighbor_key]])
                            neighbor_indices.extend(grid_ids[neighbor_key])
            
            if len(neighbor_points) < self.dbscan_min_pts:
                continue
            
            neighbor_points = np.array(neighbor_points)
            neighbor_indices = np.array(neighbor_indices)
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts)
            labels = dbscan.fit_predict(neighbor_points)
            
            # Assign cluster labels
            for i, idx in enumerate(neighbor_indices):
                if labels[i] != -1 and all_labels[idx] == -1:
                    all_labels[idx] = cluster_id + labels[i]
            
            if labels.max() != -1:
                cluster_id += labels.max() + 1
            
            processed_grids.add(grid_key)
        
        # Step 3: Compute cluster statistics
        unique_clusters = np.unique(all_labels[all_labels != -1])
        
        if len(unique_clusters) == 0:
            return np.zeros((0, 10))
        
        clustered_points = []
        
        for cluster_label in unique_clusters:
            cluster_mask = all_labels == cluster_label
            cluster_points = points[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            cluster_velocities = velocities[cluster_mask]
            cluster_rcs = rcs[cluster_mask]
            
            # Find point with minimum time difference
            min_time_idx = np.argmin(np.abs(cluster_timestamps))
            representative_point = cluster_points[min_time_idx]
            representative_velocity = cluster_velocities[min_time_idx]
            representative_rcs = cluster_rcs[min_time_idx]
            
            # Calculate cluster statistics
            num_points = cluster_points.shape[0]
            
            # Cluster density (points per cubic meter)
            # Estimate cluster volume using bounding box
            x_range = cluster_points[:, 0].max() - cluster_points[:, 0].min()
            y_range = cluster_points[:, 1].max() - cluster_points[:, 1].min()
            z_range = cluster_points[:, 2].max() - cluster_points[:, 2].min()
            
            # Avoid division by zero
            volume = max((x_range + self.grid_size) * 
                        (y_range + self.grid_size) * 
                        (z_range + self.grid_size), 
                        self.grid_size ** 3)
            
            confidence = num_points / volume  # density as confidence
            
            # Calculate mean (centroid)
            mean_point = cluster_points.mean(axis=0)
            
            # Calculate standard deviation for each dimension
            std_x = cluster_points[:, 0].std()
            std_y = cluster_points[:, 1].std()
            std_z = cluster_points[:, 2].std()
            
            # Use representative point position but add cluster statistics
            clustered_point = np.array([
                representative_point[0],  # x
                representative_point[1],  # y
                representative_point[2],  # z
                representative_velocity[0],  # vx
                representative_velocity[1],  # vy
                confidence,  # density as confidence
                std_x,  # standard deviation in x
                std_y,  # standard deviation in y
                std_z,   # standard deviation in z
                representative_rcs
            ])
            
            clustered_points.append(clustered_point)
        
        if len(clustered_points) == 0:
            return np.zeros((0, 10))
        
        return np.array(clustered_points)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) tensor
            radar_points: (N, 10) tensor [x, y, z, vx, vy, confidence, std_x, std_y, std_z, rcs]
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
        cam_ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        cam_extrinsic = transform_matrix(
            cam_calib['translation'],
            Quaternion(cam_calib['rotation']),
            inverse=False
        )
        
        # Camera ego pose transformation
        global_to_cam_ego_rot = Quaternion(cam_ego_pose['rotation']).rotation_matrix.T
        global_to_cam_ego_trans = np.array(cam_ego_pose['translation'])
        
        cam_ego_to_cam_rot = Quaternion(cam_calib['rotation']).rotation_matrix.T
        cam_ego_to_cam_trans = np.array(cam_calib['translation'])
        
        # 2. Load radar point clouds from all sweeps
        radar_points_list = []
        radar_timestamps_list = []
        radar_velocities_list = []
        radar_rcs_list = []
        image_timestamp = cam_data['timestamp']
        
        radar_sweeps = self._find_radar_sweeps(image_timestamp, self.radar_channel)
        
        print(f"Sample {idx}: Found {len(radar_sweeps)} radar sweeps")
        
        # Process each radar sweep
        for radar_data in radar_sweeps:
            radar_path = os.path.join(self.dataroot, radar_data['filename'])
            
            # Load radar points
            radar_pc = RadarPointCloud.from_file(radar_path)
            
            # Get radar calibration and ego pose
            radar_calib = self.nusc.get('calibrated_sensor', 
                                       radar_data['calibrated_sensor_token'])
            radar_ego_pose = self.nusc.get('ego_pose', radar_data['ego_pose_token'])
            
            # Transform radar points to camera frame
            points = radar_pc.points[:3, :]  # x, y, z
            velocities = radar_pc.points[8:10, :]  # vx, vy (compensated)
            rcs = radar_pc.points[5, :]  # RCS
            
            # Radar -> Radar Ego
            radar_to_ego_rot = Quaternion(radar_calib['rotation']).rotation_matrix
            radar_to_ego_trans = np.array(radar_calib['translation'])
            points_ego = radar_to_ego_rot @ points + radar_to_ego_trans.reshape(3, 1)
            
            # Radar Ego -> Global
            ego_to_global_rot = Quaternion(radar_ego_pose['rotation']).rotation_matrix
            ego_to_global_trans = np.array(radar_ego_pose['translation'])
            points_global = ego_to_global_rot @ points_ego + ego_to_global_trans.reshape(3, 1)
            
            # Global -> Camera Ego
            points_cam_ego = global_to_cam_ego_rot @ (points_global - global_to_cam_ego_trans.reshape(3, 1))
            
            # Camera Ego -> Camera
            points_cam = cam_ego_to_cam_rot @ (points_cam_ego - cam_ego_to_cam_trans.reshape(3, 1))
            
            # Filter points in front of camera
            valid_mask = points_cam[2, :] > 0
            
            if valid_mask.sum() > 0:
                points_cam = points_cam[:, valid_mask]
                velocities_cam = velocities[:, valid_mask]
                rcs_cam = rcs[valid_mask]
                
                radar_points_list.append(points_cam.T)  # (N, 3)
                radar_velocities_list.append(velocities_cam.T)  # (N, 2)
                radar_rcs_list.append(rcs_cam)  # (N,)
                
                # Timestamps (time difference from image)
                dt = (image_timestamp - radar_data['timestamp']) / 1e6  # seconds
                radar_timestamps_list.append(
                    np.full(points_cam.shape[1], dt)
                )
        
        # Concatenate all radar points
        if len(radar_points_list) > 0:
            all_points = np.vstack(radar_points_list)
            all_timestamps = np.concatenate(radar_timestamps_list)
            all_velocities = np.vstack(radar_velocities_list)
            all_rcs = np.concatenate(radar_rcs_list)
            
            print(f"  Total points before clustering: {all_points.shape[0]}")
            
            # Apply grid-based DBSCAN clustering
            clustered_radar_points = self._grid_dbscan_clustering(
                all_points, all_timestamps, all_velocities, all_rcs
            )
            
            print(f"  Clusters found: {clustered_radar_points.shape[0]}")
        else:
            clustered_radar_points = np.zeros((0, 10))
        
        # Subsample if too many clusters
        if clustered_radar_points.shape[0] > self.max_radar_points:
            indices = np.random.choice(
                clustered_radar_points.shape[0],
                self.max_radar_points,
                replace=False
            )
            clustered_radar_points = clustered_radar_points[indices]
        
        # Pad if too few clusters
        if clustered_radar_points.shape[0] < self.max_radar_points:
            padding = self.max_radar_points - clustered_radar_points.shape[0]
            clustered_radar_points = np.vstack([
                clustered_radar_points,
                np.zeros((padding, 10))
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
        radar_points_tensor = torch.from_numpy(clustered_radar_points).float()
        
        calibration = {
            'intrinsic': torch.from_numpy(cam_intrinsic_scaled).float(),
            'extrinsic': torch.from_numpy(cam_extrinsic).float()
        }
        
        return {
            'image': image_tensor,
            'radar_points': radar_points_tensor,  # Now (N, 9) with cluster statistics
            'calibration': calibration,
            'targets': targets,
            'sample_token': sample['token']
        }
    
    def _preprocess_image(self, image):
        """Resize image to target size"""
        h, w = image.shape[:2]
        target_h, target_w = self.image_size
        
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
            
            if ann['category_name'].split('.')[0] not in ['vehicle', 'human', 'movable_object']:
                continue
            
            box = self.nusc.get_box(ann_token)
            
            box.translate(-np.array(cam_extrinsic[:3, 3]))
            box.rotate(Quaternion(matrix=cam_extrinsic[:3, :3]).inverse)
            
            if box.center[2] <= 0:
                continue
            
            center_3d = box.center
            center_2d = view_points(
                center_3d.reshape(3, 1),
                cam_intrinsic,
                normalize=True
            ).T[0, :2]
            
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
                continue
            
            velocity = self.nusc.box_velocity(ann_token)[:2]
            
            boxes_3d.append({
                'center_3d': center_3d,
                'dimensions': box.wlh,
                'rotation': box.orientation.yaw_pitch_roll[0],
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