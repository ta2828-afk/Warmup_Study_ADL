"""
Main Training Script for Learning Rate Warmup Study
Orchestrates training with warmup, tracks metrics, and saves results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import json

from config import ExperimentConfig
from datasets import get_dataloaders
from models import get_model, count_parameters
from scheduler import get_scheduler
from interpretability import InterpretabilityTracker, SimpleMetricsTracker


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, 
                epoch, interpretability_tracker=None):
    """
    Train for one epoch
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Handle MedMNIST label format
        if len(target.shape) > 1:
            target = target.squeeze()
        
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Track interpretability metrics (only for first few batches of tracked epochs)
        if interpretability_tracker is not None and batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            interpretability_tracker.track_step(
                epoch=epoch,
                loss=loss.item(),
                lr=lr,
                data_batch=data[:32] if len(data) >= 32 else data,  # Use subset to save time
                target_batch=target[:32] if len(target) >= 32 else target,
                criterion=criterion
            )
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate model
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            # Handle MedMNIST label format
            if len(target.shape) > 1:
                target = target.squeeze()
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(config: ExperimentConfig):
    """
    Main training function
    
    Args:
        config: Experiment configuration
    """
    # Set seed
    set_seed(config.training.seed)
    
    # Create directories
    config.create_dirs()
    
    # Save config
    config.save(os.path.join(config.metrics_dir, 'config.yaml'))
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {config.data.name} dataset...")
    train_loader, val_loader = get_dataloaders(
        dataset_name=config.data.name,
        data_dir=config.data.data_dir,
        examples_per_class=config.data.examples_per_class,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        seed=config.training.seed
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    
    # Create model
    print(f"\nCreating {config.model.name} model...")
    model = get_model(
        config.model.name,
        num_classes=config.model.num_classes,
        use_cifar_variant=True
    )
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,}")
    
    # Create optimizer
    if config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=config.training.warmup_epochs,
        total_epochs=config.training.epochs,
        steps_per_epoch=len(train_loader),
        decay_schedule=config.training.lr_schedule,
        min_lr=0.0
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trackers
    metrics_tracker = SimpleMetricsTracker()
    
    interpretability_tracker = None
    if config.training.track_interpretability:
        interpretability_tracker = InterpretabilityTracker(
            model, 
            track_epochs=config.training.interpretability_epochs
        )
    
    # Training loop
    print(f"\nStarting training...")
    print(f"Warmup epochs: {config.training.warmup_epochs}")
    print(f"Total epochs: {config.training.epochs}")
    print(f"Optimizer: {config.training.optimizer}")
    print(f"Initial LR: {config.training.lr}")
    print(f"Batch size: {config.training.batch_size}")
    print("=" * 60)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(config.training.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            interpretability_tracker
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics_tracker.update(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr
        )
        
        # Print progress
        print(f"Epoch {epoch:3d}/{config.training.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if config.training.save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
        
        # Step scheduler
        scheduler.step()
        
        # Save metrics periodically
        if (epoch + 1) % 10 == 0:
            metrics_tracker.save(os.path.join(config.metrics_dir, 'metrics.json'))
    
    # Training complete
    training_time = time.time() - start_time
    
    print("=" * 60)
    print(f"Training complete in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {metrics_tracker.get_final_val_acc():.2f}%")
    
    # Save final metrics
    metrics_tracker.save(os.path.join(config.metrics_dir, 'metrics.json'))
    
    # Save interpretability metrics
    if interpretability_tracker is not None:
        interpretability_tracker.save(config.interpretability_dir)
        summary = interpretability_tracker.get_summary()
        print("\nInterpretability Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")
    
    # Save summary
    summary = {
        'experiment_name': config.experiment_name,
        'dataset': config.data.name,
        'examples_per_class': config.data.examples_per_class,
        'warmup_epochs': config.training.warmup_epochs,
        'optimizer': config.training.optimizer,
        'seed': config.training.seed,
        'best_val_acc': best_val_acc,
        'final_val_acc': metrics_tracker.get_final_val_acc(),
        'training_time_seconds': training_time,
        'total_epochs': config.training.epochs,
    }
    
    with open(os.path.join(config.metrics_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {config.metrics_dir}")
    
    return summary


if __name__ == '__main__':
    # Example: Run a single experiment
    from config import DataConfig, ModelConfig, TrainingConfig, ExperimentConfig
    
    config = ExperimentConfig(
        experiment_name='test_cifar10_500epc_w5_sgd',
        data=DataConfig(
            name='cifar10',
            examples_per_class=500,
            data_dir='./data'
        ),
        model=ModelConfig(
            name='resnet18',
            num_classes=10
        ),
        training=TrainingConfig(
            optimizer='sgd',
            lr=0.1,
            warmup_epochs=5,
            epochs=20,  # Reduced for testing
            batch_size=128,
            seed=42,
            track_interpretability=True
        ),
        output_dir='./results'
    )
    
    print("Running test experiment...")
    summary = train(config)
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Best accuracy: {summary['best_val_acc']:.2f}%")
