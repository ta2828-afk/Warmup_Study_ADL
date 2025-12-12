# Learning Rate Warmup Across Data Regimes

Comprehensive study examining how learning rate warmup effectiveness changes across different data regimes (50 to 5,000 examples per class).

## Project Overview

This project conducts 155 experiments across:
- **3 datasets**: CIFAR-10, CIFAR-100, MedMNIST (PathMNIST)
- **5 data regimes**: 5000, 1000, 500, 100, 50 examples per class
- **5 warmup durations**: 0, 1, 5, 10, 20 epochs
- **2 optimizers**: SGD and AdamW

**Total computational requirement**: 20-24 GPU hours

## Installation

```bash
# Clone repository or download files
cd warmup_study

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Create directory structure
bash setup_project.sh
```

## Quick Start

### 1. Run a Quick Test

Test the entire pipeline with a few short experiments:

```bash
python run_experiments.py --mode test
```

This runs 3 test experiments (one per dataset) with only 10 epochs each.

### 2. Run Full Experiments

#### Option A: Run all experiments at once

```bash
python run_experiments.py --mode all --output ./results
```

#### Option B: Run by week (as per project timeline)

```bash
# Week 1: CIFAR-10 baseline (first seed)
python run_experiments.py --mode week --week 1 --output ./results

# Week 2: CIFAR-10 full grid (remaining seeds)
python run_experiments.py --mode week --week 2 --output ./results

# Week 3: CIFAR-100, MedMNIST, and AdamW experiments
python run_experiments.py --mode week --week 3 --output ./results
```

#### Option C: Run specific range of experiments

```bash
# Run experiments 0-24
python run_experiments.py --mode range --start 0 --end 25 --output ./results
```

### 3. Run a Single Custom Experiment

```python
from config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig
from train import train

config = ExperimentConfig(
    experiment_name='my_experiment',
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
        epochs=100,
        batch_size=128,
        seed=42
    ),
    output_dir='./results'
)

summary = train(config)
print(f"Best accuracy: {summary['best_val_acc']:.2f}%")
```

## Project Structure

```
warmup_study/
├── config.py                 # Configuration management
├── datasets.py              # Dataset loading with stratified sampling
├── models.py                # ResNet-18 model definitions
├── scheduler.py             # Learning rate warmup scheduler
├── interpretability.py      # Gradient norms, curvature tracking
├── train.py                 # Main training script
├── run_experiments.py       # Batch experiment runner
├── analysis.py              # Analysis and visualization
├── requirements.txt         # Python dependencies
└── setup_project.sh         # Directory setup script

Generated during experiments:
├── data/                    # Downloaded datasets
├── results/
│   ├── checkpoints/         # Model checkpoints
│   ├── metrics/             # Training metrics (JSON)
│   ├── interpretability/    # Gradient norms, curvature
│   └── all_results.json     # Consolidated results
└── visualizations/          # Generated plots
```

## Key Features

### 1. Stratified Subsampling
Ensures balanced class distribution across all data regimes:
```python
from datasets import get_dataloaders

train_loader, val_loader = get_dataloaders(
    dataset_name='cifar10',
    examples_per_class=500,  # Creates 5,000 example subset
    batch_size=128
)
```

### 2. Learning Rate Warmup
Linear warmup from 0 to base LR, followed by cosine decay:
```python
from scheduler import WarmupScheduler

scheduler = WarmupScheduler(
    optimizer=optimizer,
    warmup_epochs=5,
    total_epochs=100,
    decay_schedule='cosine'
)
```

### 3. Interpretability Tracking
Tracks detailed metrics during first 10 epochs:
- Gradient norms (global and per-layer)
- Weight changes from initialization
- Loss curvature (approximate Hessian trace)
- Learning rate vs loss dynamics

### 4. Comprehensive Analysis
```python
from analysis import WarmupAnalyzer

analyzer = WarmupAnalyzer('./results')

# Create heatmaps
analyzer.plot_warmup_heatmap('cifar10', 'sgd', 
                            save_path='./heatmap.png')

# Find optimal warmup for each regime
optimal = analyzer.find_optimal_warmup('cifar10', 'sgd')

# Derive empirical rule
rule = analyzer.derive_empirical_rule('cifar10', 'sgd')

# Test hypotheses
hypotheses = analyzer.test_hypotheses()
```

## Experimental Design

### Data Regimes

| Config   | Examples/Class | Total  | Regime Name |
|----------|----------------|--------|-------------|
| Full     | 5,000          | 50,000 | Standard    |
| Medium   | 1,000          | 10,000 | Limited     |
| Small    | 500            | 5,000  | Low-shot    |
| Tiny     | 100            | 1,000  | Few-shot    |
| Minimal  | 50             | 500    | Extreme     |

### Experiment Grid

- **CIFAR-10**: 5×5 grid (5 regimes × 5 warmup) × 3 seeds = **75 experiments**
- **CIFAR-100**: 5×5 grid × 1 seed = **25 experiments**
- **MedMNIST**: 5×5 grid × 1 seed = **25 experiments**
- **AdamW comparisons**: **30 experiments**

**Total**: **155 experiments**

### Training Configuration

- **Model**: ResNet-18 (11.2M parameters)
- **Optimizer (SGD)**: lr=0.1, momentum=0.9, weight_decay=5e-4
- **Optimizer (AdamW)**: lr=0.001, weight_decay=5e-4
- **Batch size**: 128
- **Epochs**: 100
- **LR Schedule**: Cosine annealing after warmup

## Hypotheses

**H1**: Warmup benefit peaks at intermediate dataset sizes (500-5K examples)

**H2**: Optimal warmup duration decreases as dataset size decreases

**H3**: Patterns replicate across CIFAR-100 and MedMNIST

**H4**: AdamW shows reduced warmup sensitivity compared to SGD

## Analysis Outputs

### Heatmaps
Shows validation accuracy across data regimes and warmup durations.

### Training Curves
Loss, accuracy, and learning rate evolution for each experiment.

### Interpretability Plots
Gradient norms, weight changes, and loss curvature during warmup.

### Empirical Rule
Derived function: `optimal_warmup = f(examples_per_class)`

Example implementation:
```python
def recommended_warmup(examples_per_class):
    if examples_per_class >= 1000:
        return 5
    elif examples_per_class >= 100:
        return max(1, examples_per_class // 200)
    else:
        return 1
```

## Results Summary

After running all experiments, generate a comprehensive report:

```python
from analysis import WarmupAnalyzer

analyzer = WarmupAnalyzer('./results')
report = analyzer.create_summary_report('./analysis_results')
```

This creates:
- `summary_report.json`: All hypothesis tests and empirical rules
- Heatmaps for each dataset
- Statistical analysis

## Computational Requirements

| Component                  | Experiments | GPU Hours |
|----------------------------|-------------|-----------|
| CIFAR-10 (SGD, 3 seeds)   | 75          | 10-11     |
| CIFAR-100 (SGD, 1 seed)   | 25          | 3-4       |
| MedMNIST (SGD, 1 seed)    | 25          | 3-4       |
| AdamW experiments         | 30          | 3-4       |
| **Total**                 | **155**     | **20-24** |

### Platforms

- **Google Colab Free**: Combine with Kaggle for full 20-24 hours
- **Colab Pro**: Can complete in 1-2 sessions
- **Local GPU**: NVIDIA GPU with 6GB+ VRAM

## Timeline (4 Weeks)

| Week | Tasks                                              |
|------|----------------------------------------------------|
| 1    | Setup + CIFAR-10 baseline experiments             |
| 2    | Complete CIFAR-10 main grid (SGD, 3 seeds)        |
| 3    | CIFAR-100, MedMNIST, and AdamW experiments        |
| 4    | Analysis, visualization, and report writing       |

## Troubleshooting

### Out of Memory
Reduce batch size in config:
```python
training=TrainingConfig(batch_size=64)  # Instead of 128
```

### Slow Training
- Use fewer interpretability tracking steps
- Disable interpretability tracking for faster experiments:
```python
training=TrainingConfig(track_interpretability=False)
```

### Missing Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn tqdm medmnist --break-system-packages
```

## Citation

If you use this code, please cite:

```
@project{warmup_study_2025,
  title={Learning Rate Warmup Across Data Regimes},
  author={Toshini Agrawal},
  year={2025},
  institution={Columbia University}
}
```

## References

- Goyal et al. (2017). "Accurate, large minibatch SGD"
- He et al. (2018). "Bag of tricks for image classification"
- Kalra & Barkeshli (2024). "Why warmup the learning rate?"
- Kosson et al. (2024). "Analyzing & reducing the need for learning rate warmup"

## License

This project is for academic purposes as part of Applied Deep Learning coursework at Columbia University.
