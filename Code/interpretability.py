"""
Interpretability Metrics Tracking
Tracks gradient norms, loss curvature, weight changes during training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import json
import os


class InterpretabilityTracker:
    """
    Tracks interpretability metrics during training
    - Gradient norms (per layer and global)
    - Loss curvature (approximate Hessian trace)
    - Weight changes from initialization
    - Learning dynamics
    """
    
    def __init__(self, model: nn.Module, track_epochs: int = 10):
        """
        Args:
            model: PyTorch model to track
            track_epochs: Number of epochs to track detailed metrics
        """
        self.model = model
        self.track_epochs = track_epochs
        
        # Store initial weights
        self.initial_weights = self._get_weights_dict()
        
        # Metrics storage
        self.metrics = {
            'gradient_norms': [],
            'layer_gradient_norms': {},
            'weight_changes': [],
            'loss_curvature': [],
            'learning_rates': [],
            'losses': [],
            'epochs': []
        }
        
        # Initialize layer tracking
        for name, _ in model.named_parameters():
            if 'weight' in name:
                self.metrics['layer_gradient_norms'][name] = []
    
    def _get_weights_dict(self) -> Dict[str, torch.Tensor]:
        """Get current weights as dictionary"""
        weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data.clone().detach().cpu()
        return weights
    
    def compute_gradient_norm(self) -> float:
        """
        Compute global gradient norm (L2 norm of all gradients)
        
        Returns:
            Global gradient norm
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def compute_layer_gradient_norms(self) -> Dict[str, float]:
        """
        Compute gradient norm for each layer
        
        Returns:
            Dictionary of layer name -> gradient norm
        """
        layer_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'weight' in name:
                layer_norms[name] = param.grad.data.norm(2).item()
        return layer_norms
    
    def compute_weight_change(self) -> float:
        """
        Compute L2 distance from initial weights
        
        Returns:
            Total weight change magnitude
        """
        total_change = 0.0
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                initial = self.initial_weights[name].to(param.device)
                change = (param.data - initial).norm(2)
                total_change += change.item() ** 2
        total_change = total_change ** 0.5
        return total_change
    
    def compute_loss_curvature(self, 
                               data_batch: torch.Tensor,
                               target_batch: torch.Tensor,
                               criterion: nn.Module,
                               num_samples: int = 10) -> float:
        """
        Approximate loss curvature using finite differences
        This is a cheap approximation of the Hessian trace
        
        Args:
            data_batch: Input data
            target_batch: Target labels
            criterion: Loss function
            num_samples: Number of random directions to sample
            
        Returns:
            Approximate curvature (Hessian trace estimate)
        """
        self.model.eval()
        
        # Compute base loss
        with torch.no_grad():
            output = self.model(data_batch)
            base_loss = criterion(output, target_batch).item()
        
        # Sample random directions and compute second derivatives
        curvatures = []
        epsilon = 1e-3
        
        for _ in range(num_samples):
            # Store original parameters
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()
            
            # Sample random direction
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param)
                        noise = noise / (noise.norm() + 1e-8)  # Normalize
                        param.data.add_(noise, alpha=epsilon)
            
            # Compute perturbed loss
            with torch.no_grad():
                output_plus = self.model(data_batch)
                loss_plus = criterion(output_plus, target_batch).item()
            
            # Restore parameters
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data.copy_(original_params[name])
            
            # Estimate second derivative
            curvature = (loss_plus - base_loss) / (epsilon ** 2)
            curvatures.append(curvature)
        
        self.model.train()
        return np.mean(curvatures)
    
    def track_step(self, 
                   epoch: int, 
                   loss: float,
                   lr: float,
                   data_batch: Optional[torch.Tensor] = None,
                   target_batch: Optional[torch.Tensor] = None,
                   criterion: Optional[nn.Module] = None):
        """
        Track metrics for current step
        
        Args:
            epoch: Current epoch
            loss: Current loss value
            lr: Current learning rate
            data_batch: Optional data batch for curvature computation
            target_batch: Optional target batch for curvature computation
            criterion: Optional loss function for curvature computation
        """
        if epoch >= self.track_epochs:
            return
        
        # Always track these
        grad_norm = self.compute_gradient_norm()
        self.metrics['gradient_norms'].append(grad_norm)
        self.metrics['losses'].append(loss)
        self.metrics['learning_rates'].append(lr)
        self.metrics['epochs'].append(epoch)
        
        # Track layer-wise gradients
        layer_norms = self.compute_layer_gradient_norms()
        for name, norm in layer_norms.items():
            self.metrics['layer_gradient_norms'][name].append(norm)
        
        # Track weight changes
        weight_change = self.compute_weight_change()
        self.metrics['weight_changes'].append(weight_change)
        
        # Track curvature (expensive, so only occasionally)
        if data_batch is not None and target_batch is not None and criterion is not None:
            # Only compute every few steps to save time
            if len(self.metrics['loss_curvature']) < len(self.metrics['gradient_norms']) // 5:
                curvature = self.compute_loss_curvature(
                    data_batch, target_batch, criterion, num_samples=5
                )
                self.metrics['loss_curvature'].append(curvature)
    
    def save(self, save_dir: str):
        """Save metrics to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to JSON-serializable format
        save_dict = {
            'gradient_norms': self.metrics['gradient_norms'],
            'weight_changes': self.metrics['weight_changes'],
            'loss_curvature': self.metrics['loss_curvature'],
            'learning_rates': self.metrics['learning_rates'],
            'losses': self.metrics['losses'],
            'epochs': self.metrics['epochs'],
        }
        
        # Save main metrics
        with open(os.path.join(save_dir, 'interpretability_metrics.json'), 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        # Save layer-wise metrics separately (can be large)
        with open(os.path.join(save_dir, 'layer_gradient_norms.json'), 'w') as f:
            json.dump(self.metrics['layer_gradient_norms'], f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'mean_gradient_norm': np.mean(self.metrics['gradient_norms']),
            'max_gradient_norm': np.max(self.metrics['gradient_norms']),
            'final_weight_change': self.metrics['weight_changes'][-1] if self.metrics['weight_changes'] else 0,
            'mean_curvature': np.mean(self.metrics['loss_curvature']) if self.metrics['loss_curvature'] else 0,
        }
        return summary


class SimpleMetricsTracker:
    """Lightweight tracker for basic training metrics"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch': []
        }
    
    def update(self, epoch: int, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, lr: float):
        """Update metrics for current epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
    
    def save(self, save_path: str):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, load_path: str):
        """Load metrics from JSON file"""
        with open(load_path, 'r') as f:
            self.metrics = json.load(f)
    
    def get_best_val_acc(self) -> float:
        """Get best validation accuracy"""
        return max(self.metrics['val_acc']) if self.metrics['val_acc'] else 0.0
    
    def get_final_val_acc(self) -> float:
        """Get final validation accuracy"""
        return self.metrics['val_acc'][-1] if self.metrics['val_acc'] else 0.0


if __name__ == '__main__':
    # Test interpretability tracker
    print("Testing interpretability tracker...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create tracker
    tracker = InterpretabilityTracker(model, track_epochs=5)
    
    # Simulate training
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        # Dummy forward pass
        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))
        
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Track metrics
        tracker.track_step(
            epoch=epoch,
            loss=loss.item(),
            lr=0.1,
            data_batch=x,
            target_batch=y,
            criterion=criterion
        )
        
        # Clear gradients
        model.zero_grad()
    
    # Get summary
    summary = tracker.get_summary()
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")
    
    # Test save
    tracker.save('./test_metrics')
    print("\nMetrics saved to ./test_metrics")
    
    # Test simple tracker
    print("\n=== Testing Simple Metrics Tracker ===")
    simple_tracker = SimpleMetricsTracker()
    
    for epoch in range(10):
        simple_tracker.update(
            epoch=epoch,
            train_loss=1.0 - epoch * 0.1,
            train_acc=epoch * 0.1,
            val_loss=1.2 - epoch * 0.1,
            val_acc=epoch * 0.09,
            lr=0.1
        )
    
    print(f"Best val acc: {simple_tracker.get_best_val_acc():.4f}")
    print(f"Final val acc: {simple_tracker.get_final_val_acc():.4f}")
    
    simple_tracker.save('./test_metrics/simple_metrics.json')
    print("Simple metrics saved")
