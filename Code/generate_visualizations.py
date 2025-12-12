"""
Generate All Visualizations for Learning Rate Warmup Study
Creates all plots needed for the final report
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import WarmupAnalyzer
import argparse


def set_plot_style():
    """Set consistent plot styling"""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'


def generate_all_heatmaps(analyzer: WarmupAnalyzer, output_dir: str):
    """Generate heatmaps for all dataset-optimizer combinations"""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['cifar10', 'cifar100', 'medmnist']
    optimizers = ['sgd', 'adamw']
    metrics = ['best_val_acc', 'improvement_over_baseline']
    
    for dataset in datasets:
        for optimizer in optimizers:
            # Check if this combination exists
            data = analyzer.df[(analyzer.df['dataset'] == dataset) & 
                              (analyzer.df['optimizer'] == optimizer)]
            
            if len(data) == 0:
                continue
            
            for metric in metrics:
                if metric not in data.columns:
                    continue
                
                filename = f'heatmap_{dataset}_{optimizer}_{metric}.png'
                save_path = os.path.join(output_dir, filename)
                
                print(f"Generating {filename}...")
                analyzer.plot_warmup_heatmap(
                    dataset=dataset,
                    optimizer=optimizer,
                    metric=metric,
                    save_path=save_path
                )
                plt.close()


def generate_comparison_plots(analyzer: WarmupAnalyzer, output_dir: str):
    """Generate comparison plots across datasets and optimizers"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Warmup benefit across data regimes for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, dataset in enumerate(['cifar10', 'cifar100', 'medmnist']):
        data = analyzer.df[(analyzer.df['dataset'] == dataset) & 
                          (analyzer.df['optimizer'] == 'sgd')]
        
        if len(data) == 0:
            continue
        
        # Group by examples_per_class and warmup_epochs
        pivot_data = data.groupby(['examples_per_class', 'warmup_epochs'])['best_val_acc'].mean().reset_index()
        
        for epc in sorted(data['examples_per_class'].unique()):
            epc_data = pivot_data[pivot_data['examples_per_class'] == epc]
            axes[idx].plot(epc_data['warmup_epochs'], epc_data['best_val_acc'], 
                          marker='o', label=f'{epc} ex/class', linewidth=2)
        
        axes[idx].set_xlabel('Warmup Epochs', fontsize=12)
        axes[idx].set_ylabel('Validation Accuracy (%)', fontsize=12)
        axes[idx].set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'warmup_comparison_across_datasets.png'),
               bbox_inches='tight')
    plt.close()
    
    # Plot 2: SGD vs AdamW comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, dataset in enumerate(['cifar10', 'cifar100']):
        for optimizer in ['sgd', 'adamw']:
            data = analyzer.df[(analyzer.df['dataset'] == dataset) & 
                              (analyzer.df['optimizer'] == optimizer)]
            
            if len(data) == 0:
                continue
            
            # Average across all data regimes
            avg_by_warmup = data.groupby('warmup_epochs')['best_val_acc'].mean()
            
            axes[idx].plot(avg_by_warmup.index, avg_by_warmup.values,
                          marker='o', label=optimizer.upper(), linewidth=2)
        
        axes[idx].set_xlabel('Warmup Epochs', fontsize=12)
        axes[idx].set_ylabel('Avg Validation Accuracy (%)', fontsize=12)
        axes[idx].set_title(f'{dataset.upper()}: SGD vs AdamW', 
                           fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sgd_vs_adamw_comparison.png'),
               bbox_inches='tight')
    plt.close()


def generate_optimal_warmup_plot(analyzer: WarmupAnalyzer, output_dir: str):
    """Plot optimal warmup duration vs dataset size"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, dataset in enumerate(['cifar10', 'cifar100', 'medmnist']):
        optimal_df = analyzer.find_optimal_warmup(dataset, 'sgd')
        
        if len(optimal_df) == 0:
            continue
        
        axes[idx].plot(optimal_df['examples_per_class'], 
                      optimal_df['optimal_warmup'],
                      marker='o', linewidth=2, markersize=10, color='red')
        
        axes[idx].set_xlabel('Examples Per Class', fontsize=12)
        axes[idx].set_ylabel('Optimal Warmup (epochs)', fontsize=12)
        axes[idx].set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_xscale('log')
        axes[idx].grid(True, alpha=0.3)
        
        # Annotate points
        for _, row in optimal_df.iterrows():
            axes[idx].annotate(f"{row['optimal_warmup']:.0f}", 
                             (row['examples_per_class'], row['optimal_warmup']),
                             textcoords="offset points", xytext=(0,10), 
                             ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_warmup_vs_dataset_size.png'),
               bbox_inches='tight')
    plt.close()


def generate_improvement_plot(analyzer: WarmupAnalyzer, output_dir: str):
    """Plot improvement over baseline across regimes"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'improvement_over_baseline' not in analyzer.df.columns:
        print("Improvement metric not available, skipping...")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in ['cifar10', 'cifar100', 'medmnist']:
        data = analyzer.df[(analyzer.df['dataset'] == dataset) & 
                          (analyzer.df['optimizer'] == 'sgd') &
                          (analyzer.df['warmup_epochs'] > 0)]
        
        if len(data) == 0:
            continue
        
        avg_improvement = data.groupby('examples_per_class')['improvement_over_baseline'].mean()
        
        ax.plot(avg_improvement.index, avg_improvement.values,
               marker='o', label=dataset.upper(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Examples Per Class', fontsize=12)
    ax.set_ylabel('Avg Improvement Over No Warmup (%)', fontsize=12)
    ax.set_title('Warmup Benefit Across Data Regimes', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_across_regimes.png'),
               bbox_inches='tight')
    plt.close()


def generate_sample_training_curves(analyzer: WarmupAnalyzer, output_dir: str):
    """Generate training curves for representative experiments"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select representative experiments
    # One from each data regime for CIFAR-10 with warmup=5
    cifar10_data = analyzer.df[(analyzer.df['dataset'] == 'cifar10') & 
                               (analyzer.df['warmup_epochs'] == 5) &
                               (analyzer.df['optimizer'] == 'sgd') &
                               (analyzer.df['seed'] == 42)]
    
    for _, row in cifar10_data.iterrows():
        exp_name = row['experiment_name']
        epc = row['examples_per_class']
        
        save_path = os.path.join(output_dir, f'training_curve_{epc}epc.png')
        
        print(f"Generating training curve for {epc} examples/class...")
        analyzer.plot_training_curves(exp_name, save_path=save_path)
        plt.close()


def generate_all_visualizations(results_dir: str = './results',
                                output_dir: str = './visualizations'):
    """Generate all visualizations for the study"""
    
    print("="*70)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*70)
    
    # Set plot style
    set_plot_style()
    
    # Create analyzer
    print("\nLoading results...")
    analyzer = WarmupAnalyzer(results_dir)
    print(f"Loaded {len(analyzer.df)} experiments")
    
    # Create output directories
    heatmaps_dir = os.path.join(output_dir, 'heatmaps')
    curves_dir = os.path.join(output_dir, 'training_curves')
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    
    # Generate all visualizations
    print("\n1. Generating heatmaps...")
    generate_all_heatmaps(analyzer, heatmaps_dir)
    
    print("\n2. Generating comparison plots...")
    generate_comparison_plots(analyzer, comparisons_dir)
    
    print("\n3. Generating optimal warmup plots...")
    generate_optimal_warmup_plot(analyzer, comparisons_dir)
    
    print("\n4. Generating improvement plots...")
    generate_improvement_plot(analyzer, comparisons_dir)
    
    print("\n5. Generating sample training curves...")
    generate_sample_training_curves(analyzer, curves_dir)
    
    print("\n"+"="*70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*70)
    print(f"All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Heatmaps: {heatmaps_dir}")
    print(f"  - Training curves: {curves_dir}")
    print(f"  - Comparisons: {comparisons_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate all visualizations')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    generate_all_visualizations(args.results_dir, args.output_dir)
