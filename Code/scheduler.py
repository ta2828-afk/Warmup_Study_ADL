"""
Learning Rate Schedulers with Warmup
Implements linear warmup followed by cosine annealing or step decay
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period followed by a decay schedule
    
    Supports:
    - Linear warmup from 0 to base_lr
    - Cosine annealing after warmup
    - Step decay after warmup
    - Constant LR after warmup
    """
    
    def __init__(self, 
                 optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 decay_schedule: str = 'cosine',
                 min_lr: float = 0.0,
                 warmup_start_lr: float = 0.0,
                 steps_per_epoch: int = 1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            decay_schedule: 'cosine', 'step', or 'none'
            min_lr: Minimum learning rate (for cosine annealing)
            warmup_start_lr: Starting LR for warmup (default: 0)
            steps_per_epoch: Number of steps per epoch (for per-step scheduling)
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.decay_schedule = decay_schedule
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.steps_per_epoch = steps_per_epoch
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super(WarmupScheduler, self).__init__(optimizer)
        
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            return [self._get_warmup_lr(base_lr) for base_lr in self.base_lrs]
        else:
            # Post-warmup phase: apply decay schedule
            return [self._get_decay_lr(base_lr) for base_lr in self.base_lrs]
    
    def _get_warmup_lr(self, base_lr):
        """Calculate LR during warmup"""
        if self.warmup_epochs == 0:
            return base_lr
        
        # Linear warmup
        alpha = self.last_epoch / self.warmup_epochs
        return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
    
    def _get_decay_lr(self, base_lr):
        """Calculate LR after warmup"""
        epoch_after_warmup = self.last_epoch - self.warmup_epochs
        total_decay_epochs = self.total_epochs - self.warmup_epochs
        
        if self.decay_schedule == 'cosine':
            # Cosine annealing
            return self.min_lr + (base_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * epoch_after_warmup / total_decay_epochs)) / 2
        
        elif self.decay_schedule == 'step':
            # Step decay: reduce by 0.1 at 50% and 75% of total epochs
            if epoch_after_warmup >= 0.75 * total_decay_epochs:
                return base_lr * 0.01
            elif epoch_after_warmup >= 0.5 * total_decay_epochs:
                return base_lr * 0.1
            else:
                return base_lr
        
        elif self.decay_schedule == 'none':
            # Constant LR after warmup
            return base_lr
        
        else:
            raise ValueError(f"Unknown decay schedule: {self.decay_schedule}")


class WarmupStepScheduler(_LRScheduler):
    """
    Per-step (batch) warmup scheduler for finer-grained control
    Useful for very short warmup periods
    """
    
    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 decay_schedule: str = 'cosine',
                 min_lr: float = 0.0):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps (batches)
            total_steps: Total number of training steps
            decay_schedule: 'cosine', 'step', or 'none'
            min_lr: Minimum learning rate
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_schedule = decay_schedule
        self.min_lr = min_lr
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super(WarmupStepScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        """Calculate learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            alpha = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Decay phase
            return [self._get_decay_lr(base_lr) for base_lr in self.base_lrs]
    
    def _get_decay_lr(self, base_lr):
        """Calculate LR after warmup"""
        step_after_warmup = self.last_epoch - self.warmup_steps
        total_decay_steps = self.total_steps - self.warmup_steps
        
        if self.decay_schedule == 'cosine':
            return self.min_lr + (base_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * step_after_warmup / total_decay_steps)) / 2
        elif self.decay_schedule == 'none':
            return base_lr
        else:
            return base_lr


def get_scheduler(optimizer, 
                  warmup_epochs: int,
                  total_epochs: int,
                  steps_per_epoch: int,
                  decay_schedule: str = 'cosine',
                  min_lr: float = 0.0):
    """
    Factory function to create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        steps_per_epoch: Steps per epoch
        decay_schedule: Decay schedule after warmup
        min_lr: Minimum learning rate
        
    Returns:
        scheduler: Learning rate scheduler
    """
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        decay_schedule=decay_schedule,
        min_lr=min_lr,
        steps_per_epoch=steps_per_epoch
    )
    
    return scheduler


if __name__ == '__main__':
    # Test scheduler
    import matplotlib.pyplot as plt
    
    print("Testing learning rate schedulers...")
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Test different warmup configurations
    configs = [
        (0, 'No warmup'),
        (1, '1 epoch warmup'),
        (5, '5 epoch warmup'),
        (10, '10 epoch warmup'),
        (20, '20 epoch warmup'),
    ]
    
    total_epochs = 100
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (warmup_epochs, title) in enumerate(configs):
        if idx >= len(axes):
            break
            
        # Reset optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            decay_schedule='cosine',
            min_lr=0.0
        )
        
        lrs = []
        for epoch in range(total_epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        axes[idx].plot(lrs, linewidth=2)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Learning Rate')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvline(x=warmup_epochs, color='red', linestyle='--', 
                         alpha=0.5, label='End of warmup')
        if idx == 0:
            axes[idx].legend()
    
    # Remove empty subplot
    if len(configs) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('/home/claude/lr_schedule_test.png', dpi=150, bbox_inches='tight')
    print("Saved learning rate schedule visualization to lr_schedule_test.png")
    
    # Print sample LR values
    print("\n=== Sample Learning Rates (10 epoch warmup, 100 total epochs) ===")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=10,
        total_epochs=100,
        decay_schedule='cosine'
    )
    
    sample_epochs = [0, 1, 5, 10, 20, 50, 75, 99]
    for epoch in range(100):
        if epoch in sample_epochs:
            print(f"Epoch {epoch:3d}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
