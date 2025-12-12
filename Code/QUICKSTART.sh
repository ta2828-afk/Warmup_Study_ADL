#!/bin/bash

# Quick Start Guide for Learning Rate Warmup Study
# This script demonstrates the complete workflow

echo "=========================================="
echo "LEARNING RATE WARMUP STUDY - QUICK START"
echo "=========================================="

# Step 1: Setup
echo ""
echo "STEP 1: Setting up project structure..."
bash setup_project.sh

# Step 2: Install dependencies
echo ""
echo "STEP 2: Installing dependencies..."
echo "Run: pip install -r requirements.txt --break-system-packages"

# Step 3: Test installation
echo ""
echo "STEP 3: Testing installation..."
echo "Run: python run_experiments.py --mode test"

# Step 4: Run experiments
echo ""
echo "STEP 4: Running experiments..."
echo ""
echo "Option A - Run all experiments:"
echo "  python run_experiments.py --mode all --output ./results"
echo ""
echo "Option B - Run by week:"
echo "  Week 1: python run_experiments.py --mode week --week 1"
echo "  Week 2: python run_experiments.py --mode week --week 2"
echo "  Week 3: python run_experiments.py --mode week --week 3"
echo ""
echo "Option C - Run specific range:"
echo "  python run_experiments.py --mode range --start 0 --end 25"

# Step 5: Monitor progress
echo ""
echo "STEP 5: Monitoring progress..."
echo "Run: python utils.py --action progress"
echo "Run: python utils.py --action summary"

# Step 6: Generate visualizations
echo ""
echo "STEP 6: Generating visualizations..."
echo "Run: python generate_visualizations.py --results-dir ./results --output-dir ./visualizations"

# Step 7: Analysis
echo ""
echo "STEP 7: Running analysis..."
echo ""
cat << 'EOF'
from analysis import WarmupAnalyzer

# Load results
analyzer = WarmupAnalyzer('./results')

# Create heatmaps
analyzer.plot_warmup_heatmap('cifar10', 'sgd', save_path='./heatmap.png')

# Find optimal warmup
optimal = analyzer.find_optimal_warmup('cifar10', 'sgd')
print(optimal)

# Derive empirical rule
rule = analyzer.derive_empirical_rule('cifar10', 'sgd')
print(rule)

# Test hypotheses
hypotheses = analyzer.test_hypotheses()
print(hypotheses)

# Generate full report
report = analyzer.create_summary_report('./analysis_results')
EOF

echo ""
echo "=========================================="
echo "QUICK START GUIDE COMPLETE"
echo "=========================================="
echo ""
echo "For detailed instructions, see README.md"
