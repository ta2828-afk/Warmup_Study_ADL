"""
Checkpointed Experiment Runner for Jupyter Notebooks
Safely runs experiments with automatic checkpoint/resume capability
"""

import json
import os
from datetime import datetime
from pathlib import Path


def run_experiments_with_checkpoints(
    week=None,
    start_idx=None, 
    end_idx=None,
    output_dir='./results',
    checkpoint_dir='./experiment_checkpoints'
):
    """
    Run experiments with automatic checkpointing
    
    Args:
        week: Week number (1, 2, or 3) - uses predefined week ranges
        start_idx: Custom start index (overrides week)
        end_idx: Custom end index (overrides week)
        output_dir: Where to save experiment results
        checkpoint_dir: Where to save checkpoint files
    
    Returns:
        Summary of completed experiments
    """
    from config import generate_experiment_grid
    from train import train
    
    # Check if MedMNIST is available
    try:
        import medmnist
        MEDMNIST_AVAILABLE = True
    except ImportError:
        MEDMNIST_AVAILABLE = False
        print("⚠️  MedMNIST not available - those experiments will be skipped\n")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate all experiments
    print("Generating experiment grid...")
    all_experiments = generate_experiment_grid()
    
    # Filter out MedMNIST if not available
    if not MEDMNIST_AVAILABLE:
        original_count = len(all_experiments)
        all_experiments = [e for e in all_experiments if e.data.name != 'medmnist']
        if len(all_experiments) < original_count:
            print(f"Filtered out {original_count - len(all_experiments)} MedMNIST experiments")
    
    # Determine experiment range
    if week is not None:
        # Define week ranges
        if week == 1:
            # CIFAR-10 with first seed only (25 experiments)
            week_experiments = [e for e in all_experiments 
                              if e.data.name == 'cifar10' 
                              and e.training.optimizer == 'sgd'
                              and e.training.seed == 42]
            if week_experiments:
                start_idx = all_experiments.index(week_experiments[0])
                end_idx = start_idx + len(week_experiments)
            else:
                start_idx = 0
                end_idx = 0
            week_name = "Week 1 (CIFAR-10 baseline, seed 42, SGD)"
            
        elif week == 2:
            # Remaining CIFAR-10 seeds (50 experiments)
            week_experiments = [e for e in all_experiments 
                              if e.data.name == 'cifar10' 
                              and e.training.optimizer == 'sgd'
                              and e.training.seed in [123, 456]]
            if week_experiments:
                start_idx = all_experiments.index(week_experiments[0])
                end_idx = start_idx + len(week_experiments)
            else:
                start_idx = 0
                end_idx = 0
            week_name = "Week 2 (CIFAR-10 full grid, seeds 123/456, SGD)"
            
        elif week == 3:
            # CIFAR-100, MedMNIST, and all AdamW experiments
            week_experiments = [e for e in all_experiments 
                              if e.data.name in ['cifar100', 'medmnist']
                              or e.training.optimizer == 'adamw']
            if week_experiments:
                start_idx = all_experiments.index(week_experiments[0])
                end_idx = start_idx + len(week_experiments)
            else:
                start_idx = 0
                end_idx = 0
            week_name = "Week 3 (CIFAR-100, MedMNIST, all AdamW)"
        else:
            raise ValueError("Week must be 1, 2, or 3")
    else:
        # Custom range
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(all_experiments)
        week_name = f"Custom range ({start_idx} to {end_idx})"
    
    # Checkpoint file specific to this range
    checkpoint_name = f"checkpoint_week{week}" if week else f"checkpoint_{start_idx}_{end_idx}"
    checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
    
    # Load existing checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        completed_indices = set(checkpoint['completed_indices'])
        all_results = checkpoint['results']
        failed_experiments = checkpoint.get('failed_experiments', [])
        print(f"\n📂 Checkpoint found!")
        print(f"   Completed: {len(completed_indices)} experiments")
        print(f"   Failed: {len(failed_experiments)} experiments")
        print(f"   Last updated: {checkpoint['last_updated']}")
    else:
        completed_indices = set()
        all_results = []
        failed_experiments = []
        print(f"\n🆕 Starting fresh (no checkpoint found)")
    
    # Print summary
    total_experiments = end_idx - start_idx
    remaining = total_experiments - len(completed_indices)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT PLAN: {week_name}")
    print(f"{'='*70}")
    print(f"Range: Experiments {start_idx} to {end_idx-1}")
    print(f"Total: {total_experiments} experiments")
    print(f"Completed: {len(completed_indices)}")
    print(f"Remaining: {remaining}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"Output: {output_dir}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"{'='*70}\n")
    
    # Confirm before starting
    if remaining > 0:
        print(f"⏰ Estimated time: {remaining * 10} - {remaining * 20} minutes")
        print(f"💡 You can interrupt and resume anytime by re-running this cell\n")
    
    # Run experiments
    start_time = datetime.now()
    experiments_run_this_session = 0
    
    for idx in range(start_idx, end_idx):
        # Skip if already completed
        if idx in completed_indices:
            continue
        
        config = all_experiments[idx]
        config.output_dir = output_dir
        
        # Progress indicator
        progress = len(completed_indices) + 1
        print(f"\n{'='*70}")
        print(f"Experiment {progress}/{total_experiments} (Index: {idx})")
        print(f"{'='*70}")
        print(f"Name: {config.experiment_name}")
        print(f"Dataset: {config.data.name}")
        print(f"Examples/class: {config.data.examples_per_class} ({config.data.regime_name})")
        print(f"Warmup: {config.training.warmup_epochs} epochs")
        print(f"Optimizer: {config.training.optimizer}")
        print(f"Seed: {config.training.seed}")
        print(f"{'='*70}")
        
        try:
            # Run experiment
            summary = train(config)
            
            # Record success
            all_results.append(summary)
            completed_indices.add(idx)
            experiments_run_this_session += 1
            
            # Save checkpoint
            checkpoint = {
                'week': week,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'completed_indices': list(completed_indices),
                'results': all_results,
                'failed_experiments': failed_experiments,
                'last_updated': datetime.now().isoformat(),
                'total_experiments': total_experiments,
                'progress_percentage': (len(completed_indices) / total_experiments) * 100
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Print success
            print(f"\n✅ Experiment {idx} completed successfully!")
            print(f"   Best Val Acc: {summary['best_val_acc']:.2f}%")
            print(f"   Time: {summary['training_time_seconds']/60:.1f} minutes")
            print(f"💾 Checkpoint saved ({len(completed_indices)}/{total_experiments}, "
                  f"{checkpoint['progress_percentage']:.1f}%)")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user at experiment {idx}")
            print(f"💾 Progress saved: {len(completed_indices)}/{total_experiments} completed")
            print(f"💡 Re-run this cell to resume from experiment {idx}")
            break
            
        except Exception as e:
            print(f"\n❌ Experiment {idx} failed!")
            print(f"   Error: {str(e)}")
            
            # Record failure
            failed_experiments.append({
                'index': idx,
                'experiment_name': config.experiment_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Save checkpoint with failure
            checkpoint = {
                'week': week,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'completed_indices': list(completed_indices),
                'results': all_results,
                'failed_experiments': failed_experiments,
                'last_updated': datetime.now().isoformat(),
                'total_experiments': total_experiments,
                'progress_percentage': (len(completed_indices) / total_experiments) * 100
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            print(f"💾 Failure recorded in checkpoint")
            print(f"💡 Continuing with next experiment...")
            
            # Print traceback for debugging
            import traceback
            print("\nFull error trace:")
            traceback.print_exc()
    
    # Final summary
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"SESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Experiments run this session: {experiments_run_this_session}")
    print(f"Total completed: {len(completed_indices)}/{total_experiments}")
    print(f"Progress: {(len(completed_indices)/total_experiments)*100:.1f}%")
    print(f"Failed: {len(failed_experiments)}")
    
    if len(completed_indices) < total_experiments:
        remaining = total_experiments - len(completed_indices)
        print(f"\n⏳ Still need to complete: {remaining} experiments")
        print(f"💡 Re-run this cell to continue")
    else:
        print(f"\n🎉 All {total_experiments} experiments completed!")
        print(f"✅ Results saved to: {output_dir}")
    
    if failed_experiments:
        print(f"\n⚠️  {len(failed_experiments)} experiments failed:")
        for failure in failed_experiments[:5]:  # Show first 5
            print(f"   - {failure['experiment_name']}: {failure['error']}")
        if len(failed_experiments) > 5:
            print(f"   ... and {len(failed_experiments) - 5} more")
    
    print(f"{'='*70}\n")
    
    # Return summary
    return {
        'total': total_experiments,
        'completed': len(completed_indices),
        'failed': len(failed_experiments),
        'remaining': total_experiments - len(completed_indices),
        'progress_percentage': (len(completed_indices) / total_experiments) * 100,
        'checkpoint_file': checkpoint_file,
        'results': all_results
    }


if __name__ == '__main__':
    # Example usage
    print("This module provides checkpointed experiment running.")
    print("\nUsage:")
    print("  from run_experiments_checkpointed import run_experiments_with_checkpoints")
    print("  run_experiments_with_checkpoints(week=1)")
