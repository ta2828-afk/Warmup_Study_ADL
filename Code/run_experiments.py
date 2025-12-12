"""
Run All Experiments for Learning Rate Warmup Study
Executes all 155 experiments according to the experimental design
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

from config import generate_experiment_grid, ExperimentConfig
from train import train

# Check if MedMNIST is available
try:
    import medmnist
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("⚠️  Warning: MedMNIST not installed. MedMNIST experiments will be skipped.")
    print("   Install with: pip install medmnist --no-deps")
    print("   You will still have 124 experiments (CIFAR-10 and CIFAR-100)\n")


def run_all_experiments(start_idx=0, end_idx=None, output_dir='./results'):
    """
    Run all experiments in the grid
    
    Args:
        start_idx: Start from this experiment index (for resuming)
        end_idx: End at this experiment index (for running subset)
        output_dir: Base output directory
    """
    # Generate all experiments
    experiments = generate_experiment_grid()
    
    # Filter out MedMNIST experiments if package not available
    if not MEDMNIST_AVAILABLE:
        original_count = len(experiments)
        experiments = [e for e in experiments if e.data.name != 'medmnist']
        filtered_count = original_count - len(experiments)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} MedMNIST experiments (package not available)")
            print(f"Running {len(experiments)} experiments instead of {original_count}\n")
    
    if end_idx is None:
        end_idx = len(experiments)
    
    print(f"=" * 80)
    print(f"Learning Rate Warmup Study - Experiment Runner")
    print(f"=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Running experiments {start_idx} to {end_idx}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 80)
    
    # Track results
    all_results = []
    failed_experiments = []
    
    start_time = time.time()
    
    for idx in range(start_idx, end_idx):
        config = experiments[idx]
        config.output_dir = output_dir
        
        print(f"\n{'='*80}")
        print(f"Experiment {idx + 1}/{len(experiments)}")
        print(f"Name: {config.experiment_name}")
        print(f"Dataset: {config.data.name}")
        print(f"Examples/class: {config.data.examples_per_class} ({config.data.regime_name})")
        print(f"Warmup: {config.training.warmup_epochs} epochs")
        print(f"Optimizer: {config.training.optimizer}")
        print(f"Seed: {config.training.seed}")
        print(f"{'='*80}")
        
        try:
            # Run experiment
            summary = train(config)
            all_results.append(summary)
            
            # Save progress
            with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\n✓ Experiment {idx + 1} completed successfully")
            print(f"  Best Val Acc: {summary['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"\n✗ Experiment {idx + 1} failed with error:")
            print(f"  {str(e)}")
            failed_experiments.append({
                'index': idx,
                'experiment_name': config.experiment_name,
                'error': str(e)
            })
            
            # Save failed experiments log
            with open(os.path.join(output_dir, 'failed_experiments.json'), 'w') as f:
                json.dump(failed_experiments, f, indent=2)
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_time_per_exp = elapsed / (idx - start_idx + 1)
        remaining_exps = end_idx - idx - 1
        est_remaining = avg_time_per_exp * remaining_exps
        
        print(f"\nProgress: {idx + 1 - start_idx}/{end_idx - start_idx} experiments")
        print(f"Elapsed: {elapsed/3600:.2f} hours")
        print(f"Est. remaining: {est_remaining/3600:.2f} hours")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"All experiments complete!")
    print(f"{'='*80}")
    print(f"Total experiments run: {len(all_results)}")
    print(f"Failed experiments: {len(failed_experiments)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average time per experiment: {total_time/len(all_results)/60:.2f} minutes")
    print(f"Results saved to: {output_dir}")
    
    if failed_experiments:
        print(f"\nFailed experiments:")
        for failed in failed_experiments:
            print(f"  - {failed['experiment_name']}: {failed['error']}")


def run_week_experiments(week: int, output_dir='./results'):
    """
    Run experiments for a specific week according to the timeline
    
    Week 1: CIFAR-10 baseline (5 warmup values × 5 data regimes × 1 seed = 25)
    Week 2: CIFAR-10 main grid (5 warmup × 5 data × 3 seeds = 75)
    Week 3: CIFAR-100 (25) + MedMNIST (25) + AdamW (30) = 80
    Week 4: Analysis (no experiments)
    """
    experiments = generate_experiment_grid()
    
    # Define week ranges
    # Week 1: First seed of CIFAR-10
    # Week 2: All CIFAR-10 (seeds 2 and 3)
    # Week 3: CIFAR-100, MedMNIST, and AdamW
    
    if week == 1:
        # CIFAR-10 with first seed only (25 experiments)
        week_experiments = [e for e in experiments 
                          if e.data.name == 'cifar10' 
                          and e.training.optimizer == 'sgd'
                          and e.training.seed == 42]
        start_idx = experiments.index(week_experiments[0])
        end_idx = start_idx + len(week_experiments)
        
    elif week == 2:
        # Remaining CIFAR-10 seeds (50 experiments)
        week_experiments = [e for e in experiments 
                          if e.data.name == 'cifar10' 
                          and e.training.optimizer == 'sgd'
                          and e.training.seed != 42]
        start_idx = experiments.index(week_experiments[0])
        end_idx = start_idx + len(week_experiments)
        
    elif week == 3:
        # CIFAR-100, MedMNIST, and all AdamW experiments
        cifar10_sgd_count = sum(1 for e in experiments 
                                if e.data.name == 'cifar10' 
                                and e.training.optimizer == 'sgd')
        start_idx = cifar10_sgd_count
        end_idx = len(experiments)
        
    else:
        raise ValueError(f"Week must be 1, 2, or 3 (week 4 is analysis)")
    
    print(f"\n{'='*80}")
    print(f"Running Week {week} Experiments")
    print(f"Experiments {start_idx} to {end_idx} (total: {end_idx - start_idx})")
    print(f"{'='*80}\n")
    
    run_all_experiments(start_idx, end_idx, output_dir)


def run_quick_test(output_dir='./results_test'):
    """
    Run a quick test with a few experiments to verify everything works
    Tests one configuration from each dataset
    """
    print("Running quick test experiments...")
    
    from config import DataConfig, ModelConfig, TrainingConfig, ExperimentConfig
    
    test_configs = [
        # CIFAR-10
        ExperimentConfig(
            experiment_name='test_cifar10_500epc_w5_sgd',
            data=DataConfig(name='cifar10', examples_per_class=500),
            model=ModelConfig(num_classes=10),
            training=TrainingConfig(optimizer='sgd', warmup_epochs=5, epochs=10, seed=42),
            output_dir=output_dir
        ),
        # CIFAR-100
        ExperimentConfig(
            experiment_name='test_cifar100_500epc_w5_sgd',
            data=DataConfig(name='cifar100', examples_per_class=500),
            model=ModelConfig(num_classes=100),
            training=TrainingConfig(optimizer='sgd', warmup_epochs=5, epochs=10, seed=42),
            output_dir=output_dir
        ),
        # MedMNIST
        ExperimentConfig(
            experiment_name='test_medmnist_500epc_w5_sgd',
            data=DataConfig(name='medmnist', examples_per_class=500),
            model=ModelConfig(num_classes=9),
            training=TrainingConfig(optimizer='sgd', warmup_epochs=5, epochs=10, seed=42),
            output_dir=output_dir
        ),
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config.experiment_name}")
        try:
            summary = train(config)
            print(f"✓ Test passed - Accuracy: {summary['best_val_acc']:.2f}%")
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Learning Rate Warmup Experiments')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'week', 'test', 'range'],
                       help='Experiment mode: all, week, test, or range')
    parser.add_argument('--week', type=int, choices=[1, 2, 3],
                       help='Week number (for --mode week)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index (for --mode range)')
    parser.add_argument('--end', type=int, default=None,
                       help='End index (for --mode range)')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_all_experiments(output_dir=args.output)
    elif args.mode == 'week':
        if args.week is None:
            print("Error: --week required for week mode")
            sys.exit(1)
        run_week_experiments(args.week, output_dir=args.output)
    elif args.mode == 'test':
        run_quick_test(output_dir=args.output + '_test')
    elif args.mode == 'range':
        run_all_experiments(args.start, args.end, output_dir=args.output)
