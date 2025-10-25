import torch
import numpy as np
import os
import pickle
import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For headless environments
import json
from datetime import datetime
import seaborn as sns


class TrainingVisualizer:
    """Visualize and log training metrics"""
    
    def __init__(self, save_dir='./logs', experiment_name=None):
        """
        Args:
            save_dir: Directory to save plots and logs
            experiment_name: Name of experiment (auto-generated if None)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'plots'), exist_ok=True)
        
        # Initialize metric tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'train_box_2d_loss': [],
            'train_depth_loss': [],
            'train_vel_loss': [],
            'learning_rate': [],
            'epochs': []
        }
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print(f"Visualizer initialized. Logs will be saved to: {self.save_dir}")
    
    def log_epoch(self, epoch, train_losses, val_loss, learning_rate):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            train_losses: Dict of training losses
            val_loss: Validation loss (scalar)
            learning_rate: Current learning rate
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_losses.get('total', 0))
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_cls_loss'].append(train_losses.get('cls', 0))
        self.metrics['train_box_2d_loss'].append(train_losses.get('box_2d', 0))
        self.metrics['train_depth_loss'].append(train_losses.get('depth', 0))
        self.metrics['train_vel_loss'].append(train_losses.get('vel', 0))
        self.metrics['learning_rate'].append(learning_rate)
        
        # Save metrics to JSON
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_losses(self, save=True):
        """Plot training and validation losses"""
        if len(self.metrics['epochs']) == 0:
            print("No data to plot yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        epochs = self.metrics['epochs']
        
        # 1. Overall Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add best validation loss marker
        if self.metrics['val_loss']:
            best_val_idx = np.argmin(self.metrics['val_loss'])
            best_val_epoch = epochs[best_val_idx]
            best_val_loss = self.metrics['val_loss'][best_val_idx]
            ax.plot(best_val_epoch, best_val_loss, 'g*', markersize=15, 
                   label=f'Best Val Loss: {best_val_loss:.4f}')
            ax.legend()
        
        # 2. Component Losses
        ax = axes[0, 1]
        ax.plot(epochs, self.metrics['train_cls_loss'], label='Classification', linewidth=1.5)
        ax.plot(epochs, self.metrics['train_box_2d_loss'], label='2D Box', linewidth=1.5)
        ax.plot(epochs, self.metrics['train_depth_loss'], label='Depth', linewidth=1.5)
        ax.plot(epochs, self.metrics['train_vel_loss'], label='Velocity', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Component Losses (Training)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Learning Rate
        ax = axes[1, 0]
        ax.plot(epochs, self.metrics['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. Loss Ratio (Train/Val)
        ax = axes[1, 1]
        if len(self.metrics['val_loss']) > 0 and all(v > 0 for v in self.metrics['val_loss']):
            loss_ratio = [t/v if v > 0 else 1.0 for t, v in 
                         zip(self.metrics['train_loss'], self.metrics['val_loss'])]
            ax.plot(epochs, loss_ratio, 'purple', linewidth=2)
            ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect Ratio')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Train Loss / Val Loss')
            ax.set_title('Overfitting Indicator')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add shaded region for good generalization
            ax.fill_between(epochs, 0.8, 1.2, alpha=0.2, color='green', 
                           label='Good Generalization')
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.save_dir, 'plots', 'training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {plot_path}")
        
        plt.close()
    
    def plot_loss_components_stacked(self, save=True):
        """Plot stacked area chart of loss components"""
        if len(self.metrics['epochs']) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = self.metrics['epochs']
        
        # Prepare data
        cls_loss = np.array(self.metrics['train_cls_loss'])
        box_2d_loss = np.array(self.metrics['train_box_2d_loss'])
        depth_loss = np.array(self.metrics['train_depth_loss'])
        vel_loss = np.array(self.metrics['train_vel_loss'])
        
        # Stack plot
        ax.stackplot(epochs, cls_loss, box_2d_loss, depth_loss, vel_loss,
                    labels=['Classification', '2D Box', 'Depth', 'Velocity'],
                    alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Contribution', fontsize=12)
        ax.set_title('Loss Component Contributions Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.save_dir, 'plots', 'loss_components_stacked.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Stacked loss plot saved to: {plot_path}")
        
        plt.close()
    
    def plot_smoothed_losses(self, window=5, save=True):
        """Plot smoothed losses for better trend visibility"""
        if len(self.metrics['epochs']) < window:
            return
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = self.metrics['epochs'][window-1:]
        
        train_smooth = moving_average(self.metrics['train_loss'], window)
        val_smooth = moving_average(self.metrics['val_loss'], window)
        
        # Plot original with low alpha
        ax.plot(self.metrics['epochs'], self.metrics['train_loss'], 
               'b-', alpha=0.3, linewidth=1)
        ax.plot(self.metrics['epochs'], self.metrics['val_loss'], 
               'r-', alpha=0.3, linewidth=1)
        
        # Plot smoothed
        ax.plot(epochs, train_smooth, 'b-', label=f'Train Loss (MA-{window})', linewidth=2)
        ax.plot(epochs, val_smooth, 'r-', label=f'Val Loss (MA-{window})', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Smoothed Training Curves (Moving Average Window={window})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.save_dir, 'plots', 'smoothed_losses.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Smoothed losses saved to: {plot_path}")
        
        plt.close()
    
    def plot_summary(self):
        """Generate all plots"""
        print("\nGenerating training visualizations...")
        self.plot_losses(save=True)
        self.plot_loss_components_stacked(save=True)
        self.plot_smoothed_losses(window=5, save=True)
        print("All plots generated successfully!")
    
    def generate_report(self):
        """Generate a text summary report"""
        if len(self.metrics['epochs']) == 0:
            return
        
        report_path = os.path.join(self.save_dir, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TRAINING SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Basic info
            f.write(f"Total Epochs: {len(self.metrics['epochs'])}\n")
            f.write(f"Final Epoch: {self.metrics['epochs'][-1]}\n\n")
            
            # Loss statistics
            f.write("LOSS STATISTICS:\n")
            f.write("-"*60 + "\n")
            
            final_train = self.metrics['train_loss'][-1]
            final_val = self.metrics['val_loss'][-1]
            best_val = min(self.metrics['val_loss'])
            best_val_epoch = self.metrics['epochs'][np.argmin(self.metrics['val_loss'])]
            
            f.write(f"Final Training Loss:   {final_train:.6f}\n")
            f.write(f"Final Validation Loss: {final_val:.6f}\n")
            f.write(f"Best Validation Loss:  {best_val:.6f} (Epoch {best_val_epoch})\n\n")
            
            # Improvement
            initial_train = self.metrics['train_loss'][0]
            initial_val = self.metrics['val_loss'][0]
            train_improvement = ((initial_train - final_train) / initial_train) * 100
            val_improvement = ((initial_val - final_val) / initial_val) * 100
            
            f.write(f"Training Loss Improvement:   {train_improvement:.2f}%\n")
            f.write(f"Validation Loss Improvement: {val_improvement:.2f}%\n\n")
            
            # Component losses
            f.write("FINAL COMPONENT LOSSES:\n")
            f.write("-"*60 + "\n")
            f.write(f"Classification: {self.metrics['train_cls_loss'][-1]:.6f}\n")
            f.write(f"2D Box:         {self.metrics['train_box_2d_loss'][-1]:.6f}\n")
            f.write(f"Depth:          {self.metrics['train_depth_loss'][-1]:.6f}\n")
            f.write(f"Velocity:       {self.metrics['train_vel_loss'][-1]:.6f}\n\n")
            
            # Learning rate
            f.write("LEARNING RATE:\n")
            f.write("-"*60 + "\n")
            f.write(f"Initial: {self.metrics['learning_rate'][0]:.2e}\n")
            f.write(f"Final:   {self.metrics['learning_rate'][-1]:.2e}\n\n")
            
            # Convergence analysis
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write("-"*60 + "\n")
            
            # Check last 10 epochs
            if len(self.metrics['val_loss']) >= 10:
                recent_val = self.metrics['val_loss'][-10:]
                val_std = np.std(recent_val)
                val_trend = recent_val[-1] - recent_val[0]
                
                f.write(f"Last 10 epochs validation loss std: {val_std:.6f}\n")
                f.write(f"Last 10 epochs validation trend: {val_trend:+.6f}\n")
                
                if val_std < 0.01 and abs(val_trend) < 0.01:
                    f.write("Status: CONVERGED ✓\n")
                elif val_trend > 0.05:
                    f.write("Status: OVERFITTING ⚠\n")
                else:
                    f.write("Status: TRAINING 🔄\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Training report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Best Validation Loss: {best_val:.6f} (Epoch {best_val_epoch})")
        print(f"Final Train/Val: {final_train:.6f} / {final_val:.6f}")
        print(f"Improvement: {val_improvement:.2f}%")
        print("="*60 + "\n")
# ============================================
# UTILITY FUNCTIONS
# ============================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """Save checkpoint to file"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename):
    """Load checkpoint from file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    
    checkpoint = torch.load(filename, map_location='cpu')
    print(f"Checkpoint loaded from {filename}")
    return checkpoint


def compute_iou_3d(box1, box2):
    """
    Compute 3D IoU between two boxes
    Simplified version - implement full 3D IoU for production
    """
    # Extract centers
    c1 = box1[:3]
    c2 = box2[:3]
    
    # Compute distance
    dist = np.linalg.norm(c1 - c2)
    
    # Simple approximation based on distance
    # In practice, use proper 3D IoU computation
    if dist > 10:
        return 0.0
    else:
        return max(0, 1.0 - dist / 10.0)
    

def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    radar_points = torch.stack([item['radar_points'] for item in batch])
    radar_timestamps = torch.stack([item['radar_timestamps'] for item in batch])
    
    # Calibrations (list of dicts)
    calibrations = [item['calibration'] for item in batch]
    
    # Targets (list of dicts)
    targets = [item['targets'] for item in batch]
    
    sample_tokens = [item['sample_token'] for item in batch]
    
    return {
        'images': images,
        'radar_points': radar_points,
        'radar_timestamps': radar_timestamps,
        'calibrations': calibrations,
        'targets': targets,
        'sample_tokens': sample_tokens
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, visualizer=None):
    """Train for one epoch with detailed loss tracking"""
    from tqdm import tqdm
    
    model.train()
    
    # Track all losses
    loss_meters = {
        'total': AverageMeter(),
        'cls': AverageMeter(),
        'box_2d': AverageMeter(),
        'depth': AverageMeter(),
        'center': AverageMeter(),
        'dims': AverageMeter(),
        'rot': AverageMeter(),
        'vel': AverageMeter(),
        'uncertainty_reg': AverageMeter()
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['images'].to(device)
        radar_points = batch['radar_points'].to(device)
        radar_timestamps = batch['radar_timestamps'].to(device)
        
        # Forward pass
        predictions = model(images, radar_points, radar_timestamps, batch['calibrations'])
        
        # Compute losses
        losses = criterion(predictions, batch['targets'])
        total_loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Update meters
        batch_size = images.size(0)
        for key in loss_meters.keys():
            if key in losses:
                loss_meters[key].update(losses[key].item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meters['total'].avg:.4f}",
            'cls': f"{loss_meters['cls'].avg:.4f}",
            'depth': f"{loss_meters['depth'].avg:.4f}",
            'vel': f"{loss_meters['vel'].avg:.4f}"
        })
    
    # Return average losses
    avg_losses = {key: meter.avg for key, meter in loss_meters.items()}
    return avg_losses

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(device)
            radar_points = batch['radar_points'].to(device)
            radar_timestamps = batch['radar_timestamps'].to(device)
            
            # Forward pass
            predictions = model(
                images,
                radar_points,
                radar_timestamps,
                batch['calibrations']
            )
            
            # Compute loss
            losses = criterion(predictions, batch['targets'])
            total_loss = losses['total']
            
            loss_meter.update(total_loss.item(), images.size(0))
    
    return loss_meter.avg