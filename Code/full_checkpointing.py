"""
ULTRA-GRANULAR CHECKPOINTING
Saves after EVERY epoch, not just after complete experiments
Perfect for unstable connections or frequent interruptions
"""

import json
import os
import torch
from datetime import datetime
from pathlib import Path


def train_with_epoch_checkpoints(config, checkpoint_dir='./epoch_checkpoints'):
    """
    Training with epoch-level checkpointing
    Saves after every single epoch - can resume mid-experiment!
    
    Args:
        config: ExperimentConfig object
        checkpoint_dir: Where to save epoch checkpoints
        
    Returns:
        summary: Training summary
    """
    from train import set_seed, validate
    from datasets import get_dataloaders
    from models import get_model, count_parameters
    from scheduler import get_scheduler
    from interpretability import InterpretabilityTracker, SimpleMetricsTracker
    import torch.nn as nn
    import torch.optim as optim
    import time
    from tqdm import tqdm
    
    # Create checkpoint directory
    exp_checkpoint_dir = os.path.join(checkpoint_dir, config.experiment_name)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(exp_checkpoint_dir, 'training_checkpoint.pth')
    
    # Check for existing checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    metrics_tracker = SimpleMetricsTracker()
    
    if os.path.exists(checkpoint_file):
        print(f"📂 Found checkpoint! Resuming {config.experiment_name}...")
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        metrics_tracker.metrics = checkpoint['metrics']
        print(f"   Resuming from epoch {start_epoch}/{config.training.epochs}")
        print(f"   Best val acc so far: {best_val_acc:.2f}%")
        resume = True
    else:
        print(f"🆕 Starting {config.experiment_name} from scratch")
        resume = False
    
    # Set seed
    set_seed(config.training.seed)
    
    # Create directories
    config.create_dirs()
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    # Load data
    if not resume:
        print(f"Loading {config.data.name} dataset...")
    train_loader, val_loader = get_dataloaders(
        dataset_name=config.data.name,
        data_dir=config.data.data_dir,
        examples_per_class=config.data.examples_per_class,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        seed=config.training.seed
    )
    
    # Create model
    if not resume:
        print(f"Creating {config.model.name} model...")
    model = get_model(
        config.model.name,
        num_classes=config.model.num_classes,
        use_cifar_variant=True
    )
    model = model.to(device)
    
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
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=config.training.warmup_epochs,
        total_epochs=config.training.epochs,
        steps_per_epoch=len(train_loader),
        decay_schedule=config.training.lr_schedule,
        min_lr=0.0
    )
    
    # Load checkpoint state if resuming
    if resume:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Interpretability tracker (optional)
    interpretability_tracker = None
    if config.training.track_interpretability and not resume:
        from interpretability import InterpretabilityTracker
        interpretability_tracker = InterpretabilityTracker(
            model, 
            track_epochs=config.training.interpretability_epochs
        )
    
    # Training loop with epoch checkpoints
    print(f"\n{'='*60}")
    print(f"Training {config.experiment_name}")
    if resume:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config.training.epochs):
        # Train one epoch
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            if len(target.shape) > 1:
                target = target.squeeze()
            
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Track interpretability
            if interpretability_tracker is not None and batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                interpretability_tracker.track_step(
                    epoch=epoch,
                    loss=loss.item(),
                    lr=lr,
                    data_batch=data[:32] if len(data) >= 32 else data,
                    target_batch=target[:32] if len(target) >= 32 else target,
                    criterion=criterion
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                if len(target.shape) > 1:
                    target = target.squeeze()
                
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_loss = val_total_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
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
        
        # Update best
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        # Print progress
        status = "⭐ NEW BEST!" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{config.training.epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.2f}% | "
              f"LR: {current_lr:.6f} {status}")
        
        # SAVE CHECKPOINT AFTER EVERY EPOCH
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'metrics': metrics_tracker.metrics,
            'config': {
                'experiment_name': config.experiment_name,
                'dataset': config.data.name,
                'examples_per_class': config.data.examples_per_class,
                'warmup_epochs': config.training.warmup_epochs,
                'total_epochs': config.training.epochs,
            }
        }
        
        torch.save(checkpoint_data, checkpoint_file)
        
        if (epoch + 1) % 5 == 0 or is_best:
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
        
        # Step scheduler
        scheduler.step()
    
    # Training complete
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {metrics_tracker.get_final_val_acc():.2f}%")
    print(f"Time: {training_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Save final metrics
    metrics_tracker.save(os.path.join(config.metrics_dir, 'metrics.json'))
    
    # Save interpretability
    if interpretability_tracker is not None:
        interpretability_tracker.save(config.interpretability_dir)
    
    # Clean up epoch checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✅ Removed epoch checkpoint (training complete)")
    
    # Return summary
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
    
    return summary


def run_experiments_with_full_checkpointing(
    week=None,
    start_idx=None,
    end_idx=None,
    output_dir='./results',
    experiment_checkpoint_dir='./experiment_checkpoints',
    epoch_checkpoint_dir='./epoch_checkpoints'
):
    """
    Run experiments with both experiment-level AND epoch-level checkpointing
    
    Two-level checkpointing:
    1. Experiment level: Which experiments are complete
    2. Epoch level: Resume mid-experiment from any epoch
    
    Args:
        week: Week number (1, 2, or 3)
        start_idx: Custom start index
        end_idx: Custom end index
        output_dir: Results directory
        experiment_checkpoint_dir: Experiment-level checkpoints
        epoch_checkpoint_dir: Epoch-level checkpoints
    """
    from config import generate_experiment_grid
    
    # Check MedMNIST
    try:
        import medmnist
        MEDMNIST_AVAILABLE = True
    except ImportError:
        MEDMNIST_AVAILABLE = False
        print("⚠️  MedMNIST not available\n")
    
    # Create checkpoint directories
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)
    os.makedirs(epoch_checkpoint_dir, exist_ok=True)
    
    # Generate experiments
    all_experiments = generate_experiment_grid()
    
    # Filter MedMNIST if needed
    if not MEDMNIST_AVAILABLE:
        all_experiments = [e for e in all_experiments if e.data.name != 'medmnist']
    
    # Determine experiment range (same logic as before)
    if week is not None:
        if week == 1:
            experiments = [e for e in all_experiments 
                          if e.data.name == 'cifar10' 
                          and e.training.optimizer == 'sgd'
                          and e.training.seed == 42]
            description = "CIFAR-10 Baseline"
        elif week == 2:
            experiments = [e for e in all_experiments 
                          if e.data.name == 'cifar10' 
                          and e.training.optimizer == 'sgd'
                          and e.training.seed in [123, 456]]
            description = "CIFAR-10 Full Grid"
        elif week == 3:
            experiments = [e for e in all_experiments 
                          if e.data.name in ['cifar100', 'medmnist']
                          or e.training.optimizer == 'adamw']
            description = "Cross-dataset Validation"
        else:
            raise ValueError("Week must be 1, 2, or 3")
        
        start_idx = 0
        end_idx = len(experiments)
    else:
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(all_experiments)
        experiments = all_experiments[start_idx:end_idx]
        description = f"Custom range {start_idx}-{end_idx}"
    
    # Load experiment-level checkpoint
    exp_checkpoint_file = os.path.join(
        experiment_checkpoint_dir,
        f"{'week' + str(week) if week else 'custom'}_experiments.json"
    )
    
    if os.path.exists(exp_checkpoint_file):
        with open(exp_checkpoint_file, 'r') as f:
            exp_checkpoint = json.load(f)
        completed = set(exp_checkpoint['completed_experiments'])
        results = exp_checkpoint['results']
        print(f"📂 Experiment checkpoint: {len(completed)} completed")
    else:
        completed = set()
        results = []
        print(f"🆕 Starting {description}")
    
    total = len(experiments)
    
    print(f"\n{'='*70}")
    print(f"FULL CHECKPOINTING: {description}")
    print(f"{'='*70}")
    print(f"Total: {total} experiments")
    print(f"Completed: {len(completed)}")
    print(f"Remaining: {total - len(completed)}")
    print(f"✨ Checkpoint levels: Experiment + Epoch")
    print(f"{'='*70}\n")
    
    # Run experiments
    for i, config in enumerate(experiments):
        exp_name = config.experiment_name
        
        # Skip if completed
        if exp_name in completed:
            print(f"⏭️  [{i+1}/{total}] Skipping {exp_name} (completed)")
            continue
        
        # Update output dir
        config.output_dir = output_dir
        
        print(f"\n{'='*70}")
        print(f"[{i+1}/{total}] {exp_name}")
        print(f"{'='*70}")
        
        try:
            # Train with epoch-level checkpointing
            summary = train_with_epoch_checkpoints(
                config,
                checkpoint_dir=epoch_checkpoint_dir
            )
            
            # Record success
            results.append(summary)
            completed.add(exp_name)
            
            # Save experiment-level checkpoint
            with open(exp_checkpoint_file, 'w') as f:
                json.dump({
                    'week': week,
                    'completed_experiments': list(completed),
                    'results': results,
                    'last_updated': datetime.now().isoformat(),
                    'progress': len(completed) / total * 100
                }, f, indent=2)
            
            print(f"\n✅ {exp_name} complete: {summary['best_val_acc']:.2f}%")
            print(f"💾 Experiment checkpoint saved ({len(completed)}/{total})")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted!")
            print(f"💾 Progress saved at:")
            print(f"   Experiment level: {len(completed)}/{total} complete")
            print(f"   Epoch level: Can resume current experiment")
            print(f"💡 Re-run to continue from current state")
            break
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"⏭️  Continuing...")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {len(completed)}/{total}")
    if len(completed) < total:
        print(f"⏳ {total - len(completed)} remaining")
        print(f"💡 Re-run to continue")
    else:
        print(f"🎉 All experiments complete!")
    print(f"{'='*70}\n")
    
    return {
        'completed': len(completed),
        'total': total,
        'results': results
    }


if __name__ == '__main__':
    print("Full checkpointing module loaded!")
    print("\nFeatures:")
    print("  ✅ Saves after every epoch")
    print("  ✅ Resume mid-experiment")
    print("  ✅ Never lose progress")
