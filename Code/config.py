"""
Configuration Management for Learning Rate Warmup Study
"""

from dataclasses import dataclass
from typing import List, Optional
import yaml
import os


@dataclass
class DataConfig:
    """Dataset configuration"""
    name: str  # 'cifar10', 'cifar100', 'medmnist'
    examples_per_class: int  # 50, 100, 500, 1000, 5000
    data_dir: str = './data'
    num_workers: int = 4
    
    @property
    def total_examples(self):
        """Calculate total dataset size"""
        class_counts = {
            'cifar10': 10,
            'cifar100': 100,
            'medmnist': 9
        }
        return self.examples_per_class * class_counts.get(self.name, 10)
    
    @property
    def regime_name(self):
        """Get human-readable regime name"""
        regimes = {
            5000: 'Full',
            1000: 'Medium',
            500: 'Small',
            100: 'Tiny',
            50: 'Minimal'
        }
        return regimes.get(self.examples_per_class, 'Custom')


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = 'resnet18'
    pretrained: bool = False
    num_classes: int = 10


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimizer settings
    optimizer: str = 'sgd'  # 'sgd' or 'adamw'
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Training settings
    batch_size: int = 128
    epochs: int = 100
    warmup_epochs: int = 5
    
    # Learning rate schedule (after warmup)
    lr_schedule: str = 'cosine'  # 'cosine', 'step', or 'none'
    
    # Seed for reproducibility
    seed: int = 42
    
    # Device
    device: str = 'cuda'
    
    # Logging
    log_interval: int = 10
    save_checkpoints: bool = True
    
    # Interpretability tracking
    track_interpretability: bool = True
    interpretability_epochs: int = 10  # Track detailed metrics for first N epochs


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    
    output_dir: str = './results'
    
    def __post_init__(self):
        """Generate experiment name if not provided"""
        if self.experiment_name == 'auto':
            self.experiment_name = self._generate_name()
    
    def _generate_name(self):
        """Generate descriptive experiment name"""
        return (f"{self.data.name}_"
                f"{self.data.examples_per_class}epc_"
                f"w{self.training.warmup_epochs}_"
                f"{self.training.optimizer}_"
                f"s{self.training.seed}")
    
    @property
    def checkpoint_dir(self):
        return os.path.join(self.output_dir, 'checkpoints', self.experiment_name)
    
    @property
    def metrics_dir(self):
        return os.path.join(self.output_dir, 'metrics', self.experiment_name)
    
    @property
    def interpretability_dir(self):
        return os.path.join(self.output_dir, 'interpretability', self.experiment_name)
    
    def create_dirs(self):
        """Create all necessary directories"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        if self.training.track_interpretability:
            os.makedirs(self.interpretability_dir, exist_ok=True)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'data': {
                'name': self.data.name,
                'examples_per_class': self.data.examples_per_class,
                'data_dir': self.data.data_dir,
            },
            'model': {
                'name': self.model.name,
                'num_classes': self.model.num_classes,
            },
            'training': {
                'optimizer': self.training.optimizer,
                'lr': self.training.lr,
                'momentum': self.training.momentum,
                'weight_decay': self.training.weight_decay,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'warmup_epochs': self.training.warmup_epochs,
                'seed': self.training.seed,
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            experiment_name=config_dict['experiment_name'],
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training'])
        )


def generate_experiment_grid():
    """Generate all 155 experiments according to the proposal"""
    
    experiments = []
    
    # Dataset configurations
    datasets = [
        ('cifar10', 10),
        ('cifar100', 100),
        ('medmnist', 9)
    ]
    
    data_regimes = [5000, 1000, 500, 100, 50]
    warmup_durations = [0, 1, 5, 10, 20]
    
    # CIFAR-10: Full grid with 3 seeds (75 experiments)
    for examples_per_class in data_regimes:
        for warmup in warmup_durations:
            for seed in [42, 123, 456]:
                config = ExperimentConfig(
                    experiment_name='auto',
                    data=DataConfig(name='cifar10', examples_per_class=examples_per_class),
                    model=ModelConfig(num_classes=10),
                    training=TrainingConfig(
                        optimizer='sgd',
                        warmup_epochs=warmup,
                        seed=seed
                    )
                )
                experiments.append(config)
    
    # CIFAR-100: Full grid with 1 seed (25 experiments)
    for examples_per_class in data_regimes:
        for warmup in warmup_durations:
            config = ExperimentConfig(
                experiment_name='auto',
                data=DataConfig(name='cifar100', examples_per_class=examples_per_class),
                model=ModelConfig(num_classes=100),
                training=TrainingConfig(
                    optimizer='sgd',
                    warmup_epochs=warmup,
                    seed=42
                )
            )
            experiments.append(config)
    
    # MedMNIST: Full grid with 1 seed (25 experiments)
    for examples_per_class in data_regimes:
        for warmup in warmup_durations:
            config = ExperimentConfig(
                experiment_name='auto',
                data=DataConfig(name='medmnist', examples_per_class=examples_per_class),
                model=ModelConfig(num_classes=9),
                training=TrainingConfig(
                    optimizer='sgd',
                    warmup_epochs=warmup,
                    seed=42
                )
            )
            experiments.append(config)
    
    # AdamW comparison experiments (30 experiments)
    # Select representative configurations across different regimes
    adamw_configs = [
        (5000, 0), (5000, 5), (5000, 20),
        (1000, 0), (1000, 5), (1000, 20),
        (500, 0), (500, 5), (500, 20),
        (50, 0), (50, 1), (50, 5),
    ]
    
    for examples_per_class, warmup in adamw_configs:
        for dataset_name, num_classes in [('cifar10', 10), ('cifar100', 100)]:
            config = ExperimentConfig(
                experiment_name='auto',
                data=DataConfig(name=dataset_name, examples_per_class=examples_per_class),
                model=ModelConfig(num_classes=num_classes),
                training=TrainingConfig(
                    optimizer='adamw',
                    warmup_epochs=warmup,
                    lr=0.001,  # Lower LR for AdamW
                    seed=42
                )
            )
            experiments.append(config)
    
    # Add 6 more MedMNIST AdamW experiments to reach 30
    for examples_per_class, warmup in [(5000, 5), (1000, 5), (500, 5), (100, 1), (50, 1), (5000, 0)]:
        config = ExperimentConfig(
            experiment_name='auto',
            data=DataConfig(name='medmnist', examples_per_class=examples_per_class),
            model=ModelConfig(num_classes=9),
            training=TrainingConfig(
                optimizer='adamw',
                warmup_epochs=warmup,
                lr=0.001,
                seed=42
            )
        )
        experiments.append(config)
    
    print(f"Generated {len(experiments)} experiment configurations")
    return experiments


if __name__ == '__main__':
    # Test configuration generation
    configs = generate_experiment_grid()
    print(f"Total experiments: {len(configs)}")
    
    # Count by category
    sgd_cifar10 = sum(1 for c in configs if c.data.name == 'cifar10' and c.training.optimizer == 'sgd')
    sgd_cifar100 = sum(1 for c in configs if c.data.name == 'cifar100' and c.training.optimizer == 'sgd')
    sgd_medmnist = sum(1 for c in configs if c.data.name == 'medmnist' and c.training.optimizer == 'sgd')
    adamw_total = sum(1 for c in configs if c.training.optimizer == 'adamw')
    
    print(f"SGD CIFAR-10: {sgd_cifar10}")
    print(f"SGD CIFAR-100: {sgd_cifar100}")
    print(f"SGD MedMNIST: {sgd_medmnist}")
    print(f"AdamW total: {adamw_total}")
    
    # Save first config as example
    configs[0].create_dirs()
    configs[0].save('./example_config.yaml')
    print("\nExample config saved to example_config.yaml")
