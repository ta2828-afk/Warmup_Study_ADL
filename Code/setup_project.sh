#!/bin/bash
# Setup script for Learning Rate Warmup Study

echo "Creating project structure..."

# Create main directories
mkdir -p warmup_study/{data,models,experiments,results,analysis,visualizations,configs,logs}
mkdir -p warmup_study/results/{checkpoints,metrics,interpretability}
mkdir -p warmup_study/visualizations/{heatmaps,curves,interpretability}

echo "Project structure created!"
echo ""
echo "Directory structure:"
echo "warmup_study/"
echo "├── data/              # Datasets (auto-downloaded)"
echo "├── models/            # Model definitions"
echo "├── experiments/       # Experiment runners"
echo "├── results/           # All experimental results"
echo "│   ├── checkpoints/   # Model checkpoints"
echo "│   ├── metrics/       # Training metrics"
echo "│   └── interpretability/ # Gradient norms, curvature, etc."
echo "├── analysis/          # Analysis and plotting scripts"
echo "├── visualizations/    # Generated plots"
echo "├── configs/           # Experiment configurations"
echo "└── logs/              # Training logs"
