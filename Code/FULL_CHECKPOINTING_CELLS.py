# ====================================================================
# FULL CHECKPOINTING - JUPYTER CELLS
# Saves after EVERY epoch - Resume from anywhere!
# ====================================================================

# ====================================================================
# CELL 1: Import Full Checkpointing
# ====================================================================
from full_checkpointing import (
    train_with_epoch_checkpoints,
    run_experiments_with_full_checkpointing
)

print("✅ Full checkpointing loaded!")
print("\nThis version saves:")
print("  💾 After every epoch (can resume mid-experiment)")
print("  💾 After every experiment (can resume mid-week)")
print("\n🛡️ Maximum protection from interruptions!")


# ====================================================================
# CELL 2: Single Experiment with Epoch Checkpoints
# ====================================================================
# Test with one experiment - saves after every epoch!

from config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    experiment_name='test_epoch_checkpointing',
    data=DataConfig(name='cifar10', examples_per_class=500),
    model=ModelConfig(num_classes=10),
    training=TrainingConfig(
        optimizer='sgd',
        warmup_epochs=5,
        epochs=30,  # Will checkpoint after each of 30 epochs
        batch_size=128,
        seed=42
    ),
    output_dir='./results'
)

# This will save after EVERY single epoch
summary = train_with_epoch_checkpoints(config)

print(f"\n✅ Complete! Best accuracy: {summary['best_val_acc']:.2f}%")


# ====================================================================
# CELL 3: Week 1 with Full Checkpointing
# ====================================================================
# Runs 25 experiments, each with epoch-level checkpoints

print("Starting Week 1 with full checkpointing...")
print("Each experiment saves after every epoch!")
print("You can stop and resume at ANY point!\n")

summary = run_experiments_with_full_checkpointing(
    week=1,
    output_dir='./results',
    experiment_checkpoint_dir='./experiment_checkpoints',
    epoch_checkpoint_dir='./epoch_checkpoints'
)

print(f"\nWeek 1: {summary['completed']}/{summary['total']} experiments")


# ====================================================================
# CELL 4: Week 2 with Full Checkpointing
# ====================================================================
print("Starting Week 2 with full checkpointing...")

summary = run_experiments_with_full_checkpointing(
    week=2,
    output_dir='./results',
    experiment_checkpoint_dir='./experiment_checkpoints',
    epoch_checkpoint_dir='./epoch_checkpoints'
)

print(f"\nWeek 2: {summary['completed']}/{summary['total']} experiments")


# ====================================================================
# CELL 5: Week 3 with Full Checkpointing
# ====================================================================
print("Starting Week 3 with full checkpointing...")
print("Includes: CIFAR-100 + MedMNIST + AdamW\n")

summary = run_experiments_with_full_checkpointing(
    week=3,
    output_dir='./results',
    experiment_checkpoint_dir='./experiment_checkpoints',
    epoch_checkpoint_dir='./epoch_checkpoints'
)

print(f"\nWeek 3: {summary['completed']}/{summary['total']} experiments")


# ====================================================================
# CELL 6: Check Checkpoint Status
# ====================================================================
import os
import json

print("="*70)
print("CHECKPOINT STATUS")
print("="*70)

# Check experiment-level checkpoints
exp_checkpoint_dir = './experiment_checkpoints'
for week in [1, 2, 3]:
    checkpoint_file = os.path.join(exp_checkpoint_dir, f'week{week}_experiments.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f"\nWeek {week} (Experiment Level):")
        print(f"  Completed: {len(data['completed_experiments'])} experiments")
        print(f"  Progress: {data['progress']:.1f}%")

# Check epoch-level checkpoints
epoch_checkpoint_dir = './epoch_checkpoints'
if os.path.exists(epoch_checkpoint_dir):
    epoch_checkpoints = [d for d in os.listdir(epoch_checkpoint_dir) 
                        if os.path.isdir(os.path.join(epoch_checkpoint_dir, d))]
    
    if epoch_checkpoints:
        print(f"\n📂 Epoch-level checkpoints found:")
        for exp_dir in epoch_checkpoints[:5]:  # Show first 5
            checkpoint_path = os.path.join(epoch_checkpoint_dir, exp_dir, 'training_checkpoint.pth')
            if os.path.exists(checkpoint_path):
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                epoch = checkpoint['epoch'] + 1
                total_epochs = checkpoint['config']['total_epochs']
                best_acc = checkpoint['best_val_acc']
                print(f"  • {exp_dir}")
                print(f"    Progress: Epoch {epoch}/{total_epochs} ({epoch/total_epochs*100:.1f}%)")
                print(f"    Best acc: {best_acc:.2f}%")
    else:
        print(f"\n✅ No in-progress experiments (all complete or not started)")


# ====================================================================
# CELL 7: Resume Interrupted Experiment
# ====================================================================
# If an experiment was interrupted mid-training, it will auto-resume!

# Just re-run the week cell (Cell 3, 4, or 5)
# It will:
# 1. Skip completed experiments
# 2. Resume the interrupted one from its last epoch
# 3. Continue with remaining experiments

print("To resume:")
print("  1. Re-run the week cell (Cell 3, 4, or 5)")
print("  2. It automatically detects and resumes interrupted experiments")
print("  3. No manual intervention needed!")


# ====================================================================
# CELL 8: Example - What Happens on Interruption
# ====================================================================
"""
SCENARIO: Training interrupted during Experiment 12, Epoch 18/30

What gets saved:
  ✅ Experiments 1-11: Fully complete
  ✅ Experiment 12: 
     - Model weights at epoch 17
     - Optimizer state at epoch 17
     - All metrics up to epoch 17
     - Can resume from epoch 18

What happens when you re-run:
  ⏭️ Skips experiments 1-11 (already done)
  ▶️ Loads experiment 12 checkpoint
  ▶️ Resumes from epoch 18/30
  ▶️ Continues to experiments 13, 14, 15...

Result:
  🎉 NO WORK LOST!
  🎉 Resume exactly where you stopped
  🎉 Even mid-experiment!
"""


# ====================================================================
# CELL 9: Clean Up Checkpoints (After Week Complete)
# ====================================================================
# Optional: Remove epoch checkpoints after week is complete

import shutil
import os

def cleanup_epoch_checkpoints():
    """Remove epoch checkpoints after successful completion"""
    epoch_dir = './epoch_checkpoints'
    if os.path.exists(epoch_dir):
        size_mb = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(epoch_dir)
            for filename in filenames
        ) / (1024 * 1024)
        
        response = input(f"Remove {size_mb:.1f} MB of epoch checkpoints? (yes/no): ")
        if response.lower() == 'yes':
            shutil.rmtree(epoch_dir)
            print("✅ Epoch checkpoints removed")
        else:
            print("❌ Kept epoch checkpoints")
    else:
        print("No epoch checkpoints to clean")

# Uncomment to run:
# cleanup_epoch_checkpoints()


# ====================================================================
# CELL 10: Quick Test - 5 Epoch Run with Checkpoints
# ====================================================================
# Test on a tiny 5-epoch experiment to see checkpointing in action

from config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig

print("Running 5-epoch test with checkpointing...")
print("Watch for '💾 Checkpoint saved' messages after each epoch!\n")

config = ExperimentConfig(
    experiment_name='quick_5epoch_test',
    data=DataConfig(name='cifar10', examples_per_class=100),
    model=ModelConfig(num_classes=10),
    training=TrainingConfig(
        optimizer='sgd',
        warmup_epochs=0,
        epochs=5,  # Just 5 epochs - saves after each!
        batch_size=128,
        seed=42,
        track_interpretability=False  # Faster
    )
)

summary = train_with_epoch_checkpoints(config)

print(f"\n✅ Test complete!")
print(f"   Accuracy: {summary['best_val_acc']:.2f}%")
print(f"\nTry interrupting (Ctrl+C) during training, then re-run!")
print(f"It will resume from the last completed epoch!")


# ====================================================================
# CELL 11: Manual Resume Example
# ====================================================================
# Example: Manually resume a specific experiment

from config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig

# Create the SAME config as before
config = ExperimentConfig(
    experiment_name='quick_5epoch_test',  # Same name as interrupted experiment
    data=DataConfig(name='cifar10', examples_per_class=100),
    model=ModelConfig(num_classes=10),
    training=TrainingConfig(
        optimizer='sgd',
        warmup_epochs=0,
        epochs=5,
        batch_size=128,
        seed=42
    )
)

# Just call train again - it automatically resumes!
summary = train_with_epoch_checkpoints(config)

print(f"Resumed and completed: {summary['best_val_acc']:.2f}%")


# ====================================================================
# CELL 12: Comparison - With vs Without Checkpointing
# ====================================================================
"""
WITHOUT Epoch Checkpointing:
  Training 30 epochs...
  Epoch 1 ✓
  Epoch 2 ✓
  ...
  Epoch 18 ✓
  [CRASH] 💥
  ❌ Lost 18 epochs of work
  ❌ Must restart from epoch 1

WITH Epoch Checkpointing:
  Training 30 epochs...
  Epoch 1 ✓ 💾
  Epoch 2 ✓ 💾
  ...
  Epoch 18 ✓ 💾
  [CRASH] 💥
  Re-run...
  📂 Found checkpoint at epoch 18
  ▶️ Resuming from epoch 19
  Epoch 19 ✓ 💾
  ...
  ✅ Complete!
"""


# ====================================================================
# SUMMARY
# ====================================================================
"""
FULL CHECKPOINTING FEATURES:

✅ Saves after EVERY epoch
✅ Saves after EVERY experiment
✅ Resume mid-experiment
✅ Resume mid-week
✅ Works with all datasets
✅ Works with all optimizers
✅ Automatic - no manual intervention
✅ Small checkpoint files
✅ Can clean up when done

PROTECTION LEVELS:
🛡️ Level 1: Experiment checkpoints (which experiments done)
🛡️ Level 2: Epoch checkpoints (which epochs done)
🛡️ Result: Can resume from ANY point!

USAGE:
Just run the week cells (3, 4, 5)
Everything else is automatic!
"""
