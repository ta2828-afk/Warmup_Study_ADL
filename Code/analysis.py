"""
Analysis and Visualization for Learning Rate Warmup Study
Creates heatmaps, training curves, and derives empirical rules
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class WarmupAnalyzer:
    """Analyze warmup experiment results"""
    
    def __init__(self, results_dir: str = './results'):
        """
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.all_results = self.load_all_results()
        self.df = self.create_dataframe()
    
    def load_all_results(self) -> List[Dict]:
        """Load all experiment results"""
        all_results_path = os.path.join(self.results_dir, 'all_results.json')
        
        if os.path.exists(all_results_path):
            with open(all_results_path, 'r') as f:
                return json.load(f)
        else:
            # Load from individual experiment directories
            results = []
            metrics_dir = os.path.join(self.results_dir, 'metrics')
            
            if not os.path.exists(metrics_dir):
                print(f"Warning: No results found in {self.results_dir}")
                return []
            
            for exp_dir in os.listdir(metrics_dir):
                summary_path = os.path.join(metrics_dir, exp_dir, 'summary.json')
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        results.append(json.load(f))
            
            return results
    
    def create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        if not self.all_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.all_results)
        
        # Add derived columns
        if 'best_val_acc' in df.columns and 'warmup_epochs' in df.columns:
            # Calculate improvement over no warmup for each configuration
            for dataset in df['dataset'].unique():
                for epc in df['examples_per_class'].unique():
                    for optimizer in df['optimizer'].unique():
                        for seed in df['seed'].unique():
                            mask = ((df['dataset'] == dataset) & 
                                   (df['examples_per_class'] == epc) &
                                   (df['optimizer'] == optimizer) &
                                   (df['seed'] == seed))
                            
                            if mask.sum() > 0:
                                baseline_mask = mask & (df['warmup_epochs'] == 0)
                                if baseline_mask.sum() > 0:
                                    baseline_acc = df.loc[baseline_mask, 'best_val_acc'].values[0]
                                    df.loc[mask, 'improvement_over_baseline'] = \
                                        df.loc[mask, 'best_val_acc'] - baseline_acc
        
        return df
    
    def plot_warmup_heatmap(self, dataset: str = 'cifar10', optimizer: str = 'sgd',
                           metric: str = 'best_val_acc', save_path: str = None):
        """
        Create heatmap showing warmup effect across data regimes
        
        Args:
            dataset: Dataset name
            optimizer: Optimizer name
            metric: Metric to plot ('best_val_acc' or 'improvement_over_baseline')
            save_path: Path to save figure
        """
        # Filter data
        data = self.df[(self.df['dataset'] == dataset) & 
                       (self.df['optimizer'] == optimizer)]
        
        if len(data) == 0:
            print(f"No data found for {dataset} with {optimizer}")
            return
        
        # Aggregate across seeds (mean)
        pivot_data = data.groupby(['examples_per_class', 'warmup_epochs'])[metric].mean().reset_index()
        pivot = pivot_data.pivot(index='examples_per_class', 
                                columns='warmup_epochs', 
                                values=metric)
        
        # Sort by examples_per_class
        pivot = pivot.sort_index(ascending=False)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        
        if metric == 'improvement_over_baseline':
            cmap = 'RdYlGn'
            center = 0
        else:
            cmap = 'viridis'
            center = None
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, center=center,
                   cbar_kws={'label': metric.replace('_', ' ').title()})
        
        plt.title(f'Warmup Effect: {dataset.upper()} ({optimizer.upper()})',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Warmup Epochs', fontsize=12)
        plt.ylabel('Examples Per Class', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        else:
            plt.show()
    
    def plot_training_curves(self, experiment_name: str, save_path: str = None):
        """Plot training curves for a specific experiment"""
        metrics_path = os.path.join(self.results_dir, 'metrics', 
                                   experiment_name, 'metrics.json')
        
        if not os.path.exists(metrics_path):
            print(f"Metrics not found for {experiment_name}")
            return
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(metrics['epoch'], metrics['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(metrics['epoch'], metrics['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(metrics['epoch'], metrics['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(metrics['epoch'], metrics['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(metrics['epoch'], metrics['learning_rate'], linewidth=2, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Val accuracy zoomed on warmup period
        if len(metrics['epoch']) >= 20:
            warmup_epochs = 20
            axes[1, 1].plot(metrics['epoch'][:warmup_epochs], 
                          metrics['val_acc'][:warmup_epochs], 
                          linewidth=2, color='green', marker='o')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Accuracy (%)')
            axes[1, 1].set_title('Validation Accuracy (First 20 Epochs)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Curves: {experiment_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()
    
    def plot_interpretability_metrics(self, experiment_name: str, save_path: str = None):
        """Plot interpretability metrics for a specific experiment"""
        interp_path = os.path.join(self.results_dir, 'interpretability',
                                  experiment_name, 'interpretability_metrics.json')
        
        if not os.path.exists(interp_path):
            print(f"Interpretability metrics not found for {experiment_name}")
            return
        
        with open(interp_path, 'r') as f:
            metrics = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gradient norms
        axes[0, 0].plot(metrics['gradient_norms'], linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_title('Gradient Norms')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weight changes
        axes[0, 1].plot(metrics['weight_changes'], linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Weight Change')
        axes[0, 1].set_title('Weight Changes from Initialization')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss curvature
        if len(metrics['loss_curvature']) > 0:
            axes[1, 0].plot(metrics['loss_curvature'], linewidth=2, color='red')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Curvature')
            axes[1, 0].set_title('Loss Curvature (Hessian Approximation)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate vs loss
        axes[1, 1].scatter(metrics['learning_rates'], metrics['losses'], alpha=0.5)
        axes[1, 1].set_xlabel('Learning Rate')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Learning Rate vs Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Interpretability Metrics: {experiment_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved interpretability plots to {save_path}")
        else:
            plt.show()
    
    def find_optimal_warmup(self, dataset: str = 'cifar10', 
                           optimizer: str = 'sgd') -> pd.DataFrame:
        """
        Find optimal warmup duration for each data regime
        
        Returns:
            DataFrame with optimal warmup for each examples_per_class
        """
        data = self.df[(self.df['dataset'] == dataset) & 
                       (self.df['optimizer'] == optimizer)]
        
        # Group by examples_per_class and find best warmup
        optimal = []
        
        for epc in data['examples_per_class'].unique():
            epc_data = data[data['examples_per_class'] == epc]
            
            # Average across seeds
            avg_by_warmup = epc_data.groupby('warmup_epochs')['best_val_acc'].mean()
            
            optimal_warmup = avg_by_warmup.idxmax()
            optimal_acc = avg_by_warmup.max()
            baseline_acc = avg_by_warmup[0] if 0 in avg_by_warmup.index else None
            
            optimal.append({
                'examples_per_class': epc,
                'optimal_warmup': optimal_warmup,
                'optimal_acc': optimal_acc,
                'baseline_acc': baseline_acc,
                'improvement': optimal_acc - baseline_acc if baseline_acc else None
            })
        
        return pd.DataFrame(optimal).sort_values('examples_per_class', ascending=False)
    
    def derive_empirical_rule(self, dataset: str = 'cifar10',
                             optimizer: str = 'sgd') -> Dict:
        """
        Derive empirical rule for warmup duration based on dataset size
        
        Returns:
            Dictionary with rule parameters and evaluation metrics
        """
        optimal_df = self.find_optimal_warmup(dataset, optimizer)
        
        X = optimal_df['examples_per_class'].values.reshape(-1, 1)
        y = optimal_df['optimal_warmup'].values
        
        # Try different models
        models = {}
        
        # Linear model
        lr = LinearRegression()
        lr.fit(np.log(X + 1), y)
        models['log_linear'] = {
            'model': lr,
            'predictions': lr.predict(np.log(X + 1)),
            'r2': lr.score(np.log(X + 1), y)
        }
        
        # Piecewise function
        def piecewise_warmup(epc):
            """Derived piecewise function"""
            if epc >= 1000:
                return 5
            elif epc >= 100:
                return max(1, int(epc // 200))
            else:
                return 1
        
        piecewise_pred = np.array([piecewise_warmup(epc) for epc in X.flatten()])
        models['piecewise'] = {
            'predictions': piecewise_pred,
            'mae': np.mean(np.abs(y - piecewise_pred))
        }
        
        return {
            'optimal_values': optimal_df.to_dict('records'),
            'models': models,
            'recommendation': 'Use piecewise function for practical applications'
        }
    
    def test_hypotheses(self) -> Dict:
        """
        Test the four hypotheses from the proposal
        
        H1: Warmup benefit peaks at intermediate dataset sizes
        H2: Optimal warmup duration decreases as dataset size decreases  
        H3: Patterns replicate across CIFAR-100 and MedMNIST
        H4: AdamW shows reduced warmup sensitivity
        """
        results = {}
        
        # H1: Test if improvement peaks at intermediate sizes
        sgd_data = self.df[self.df['optimizer'] == 'sgd']
        
        if 'improvement_over_baseline' in sgd_data.columns:
            avg_improvement = sgd_data.groupby('examples_per_class')['improvement_over_baseline'].mean()
            peak_size = avg_improvement.idxmax()
            
            results['H1'] = {
                'peak_dataset_size': int(peak_size),
                'peak_improvement': float(avg_improvement.max()),
                'conclusion': f'Peak benefit at {peak_size} examples/class'
            }
        
        # H2: Test if optimal warmup decreases with dataset size
        optimal_cifar10 = self.find_optimal_warmup('cifar10', 'sgd')
        correlation = stats.spearmanr(optimal_cifar10['examples_per_class'],
                                     optimal_cifar10['optimal_warmup'])
        
        results['H2'] = {
            'correlation': float(correlation.correlation),
            'p_value': float(correlation.pvalue),
            'conclusion': 'Positive correlation' if correlation.correlation > 0 else 'Negative correlation'
        }
        
        # H3: Test cross-dataset consistency
        optimal_c10 = self.find_optimal_warmup('cifar10', 'sgd')
        optimal_c100 = self.find_optimal_warmup('cifar100', 'sgd')
        optimal_med = self.find_optimal_warmup('medmnist', 'sgd')
        
        # Compare optimal warmup values
        merged = optimal_c10.merge(optimal_c100, on='examples_per_class', suffixes=('_c10', '_c100'))
        consistency = np.mean(merged['optimal_warmup_c10'] == merged['optimal_warmup_c100'])
        
        results['H3'] = {
            'consistency_c10_c100': float(consistency),
            'conclusion': f'{consistency*100:.1f}% agreement between CIFAR-10 and CIFAR-100'
        }
        
        # H4: Test AdamW vs SGD sensitivity
        sgd_std = sgd_data.groupby('examples_per_class')['best_val_acc'].std().mean()
        adamw_data = self.df[self.df['optimizer'] == 'adamw']
        
        if len(adamw_data) > 0:
            adamw_std = adamw_data.groupby('examples_per_class')['best_val_acc'].std().mean()
            
            results['H4'] = {
                'sgd_std': float(sgd_std),
                'adamw_std': float(adamw_std),
                'conclusion': 'AdamW less sensitive' if adamw_std < sgd_std else 'SGD less sensitive'
            }
        
        return results
    
    def create_summary_report(self, save_dir: str):
        """Create comprehensive summary report"""
        os.makedirs(save_dir, exist_ok=True)
        
        report = {
            'total_experiments': len(self.df),
            'datasets': list(self.df['dataset'].unique()),
            'optimizers': list(self.df['optimizer'].unique()),
            'data_regimes': sorted(self.df['examples_per_class'].unique()),
            'warmup_values': sorted(self.df['warmup_epochs'].unique()),
            'hypothesis_tests': self.test_hypotheses(),
        }
        
        # Add empirical rules for each dataset
        report['empirical_rules'] = {}
        for dataset in ['cifar10', 'cifar100', 'medmnist']:
            if dataset in self.df['dataset'].values:
                report['empirical_rules'][dataset] = self.derive_empirical_rule(dataset, 'sgd')
        
        # Save report
        with open(os.path.join(save_dir, 'summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Summary report saved to {save_dir}/summary_report.json")
        
        return report


if __name__ == '__main__':
    # Example usage
    print("Testing analysis module...")
    
    # Note: This requires actual results to exist
    # For now, we'll create dummy results for testing
    
    # Create dummy results
    dummy_results = []
    for dataset, num_classes in [('cifar10', 10), ('cifar100', 100)]:
        for epc in [5000, 1000, 500, 100, 50]:
            for warmup in [0, 1, 5, 10, 20]:
                for seed in [42]:
                    dummy_results.append({
                        'experiment_name': f'{dataset}_{epc}epc_w{warmup}_sgd_s{seed}',
                        'dataset': dataset,
                        'examples_per_class': epc,
                        'warmup_epochs': warmup,
                        'optimizer': 'sgd',
                        'seed': seed,
                        'best_val_acc': 70 + np.random.rand() * 20 + (warmup > 0) * 2,
                        'final_val_acc': 68 + np.random.rand() * 20,
                    })
    
    # Save dummy results
    os.makedirs('./test_results', exist_ok=True)
    with open('./test_results/all_results.json', 'w') as f:
        json.dump(dummy_results, f)
    
    # Test analyzer
    analyzer = WarmupAnalyzer('./test_results')
    print(f"Loaded {len(analyzer.df)} experiments")
    
    # Test heatmap
    analyzer.plot_warmup_heatmap('cifar10', 'sgd', 
                                save_path='./test_heatmap.png')
    
    # Test optimal warmup finding
    optimal = analyzer.find_optimal_warmup('cifar10', 'sgd')
    print("\nOptimal warmup durations:")
    print(optimal)
    
    # Test empirical rule
    rule = analyzer.derive_empirical_rule('cifar10', 'sgd')
    print("\nEmpirical rule derived")
    
    print("\nAnalysis module test complete!")
