# Advanced Plotting for Backdoor Attack Analysis

This document provides comprehensive plotting functionality for your Unified Backdoor Attack and Defense Framework. The plotting code generates four key visualizations as requested.

## Files Created

1. **`evaluation/advanced_plots.py`** - Main plotting functions
2. **`generate_plots.py`** - Integration with your existing framework
3. **`plot_examples.py`** - Examples using your existing results
4. **`plotting_requirements.txt`** - Compatible dependency versions

## Plot Types Generated

### 1. Model Performance vs Poisoning Rate (0-20%, 5% steps)

**Function:** `plot_poisoning_rate_performance()`

Generates 4 subplots:
- **Clean Accuracy vs Poisoning Rate** - Shows how accuracy degrades
- **Attack Success Rate vs Poisoning Rate** - Shows attack effectiveness
- **Model Robustness Index** - Combined metric (ACC - 0.5*ASR)
- **Performance Drop Rate** - Percentage degradation from baseline

**Attack Methods Supported:** BadNet, Blend, Dynamic, SIG, WaNet

```python
from evaluation.advanced_plots import AdvancedPlots

plotter = AdvancedPlots()
plotter.plot_poisoning_rate_performance(
    poisoning_rates=[0, 5, 10, 15, 20],
    attack_methods=['badnet', 'blend', 'dynamic', 'sig', 'wanet'],
    results_data=your_results_dict,  # Optional: uses synthetic data if None
    save_path='figures/poisoning_analysis.png'
)
```

### 2. Feature Space Visualization for ResNet-18 on CIFAR-10

**Function:** `visualize_feature_space()`

Generates 4 subplots:
- **Clean vs Infected Samples** - t-SNE/PCA projection
- **Clean Samples by Class** - Class distribution in feature space
- **Clean Sample Density** - Hexbin density plot
- **Infected Sample Density** - Hexbin density plot

```python
# Requires actual model and data loaders
plotter.visualize_feature_space(
    model=your_resnet18_model,
    clean_loader=clean_test_loader,
    infected_loader=infected_test_loader,
    method='tsne',  # or 'pca'
    save_path='figures/feature_visualization.png'
)
```

### 3. Finetuning vs ASR and ACC

**Function:** `plot_finetuning_analysis()`

Generates 4 subplots:
- **ASR vs Finetuning Epochs** - Attack success rate over finetuning
- **ACC vs Finetuning Epochs** - Clean accuracy over finetuning
- **ASR vs ACC Trade-off** - Scatter plot with progression arrows
- **Defense Effectiveness** - Combined score (ACC - ASR)

**Defense Methods:** ABL, AIBD, CBD, DBD, NAD

```python
finetuning_epochs = [0, 5, 10, 15, 20, 25, 30]
asr_data = {
    'abl': [95, 80, 60, 40, 25, 15, 10],
    'aibd': [90, 70, 50, 30, 20, 12, 8],
    # ... more methods
}
acc_data = {
    'abl': [70, 75, 80, 83, 85, 86, 87],
    'aibd': [72, 77, 82, 85, 87, 88, 89],
    # ... more methods
}

plotter.plot_finetuning_analysis(finetuning_epochs, asr_data, acc_data)
```

### 4. Training Loss vs Epochs (Clean and Backdoor)

**Function:** `plot_training_loss_analysis()`

Generates 4 subplots:
- **Training Loss Comparison** - Clean vs backdoor loss curves
- **Loss Difference Analysis** - Backdoor - Clean loss over time
- **Validation Accuracy** - Accuracy curves during training
- **Learning Curves Summary** - Normalized loss comparison

```python
training_history = {
    'baseline': {
        'clean_loss': [2.3, 1.8, 1.2, 0.8, ...],
        'backdoor_loss': [1.9, 1.4, 1.0, 0.6, ...],
        'val_acc': [60, 65, 72, 78, ...]
    },
    'abl_defense': {
        'clean_loss': [2.5, 2.0, 1.5, 1.1, ...],
        'backdoor_loss': [2.2, 1.7, 1.3, 0.9, ...],
        'val_acc': [58, 63, 70, 76, ...]
    }
    # ... more methods
}

plotter.plot_training_loss_analysis(training_history)
```

## Usage Examples

### Quick Demo with Synthetic Data

```python
# Run this for immediate demonstration
python plot_examples.py --demo
```

### Integration with Existing Results

```python
# Use your existing experimental results
python plot_examples.py
```

### Full Analysis Pipeline

```python
# Run comprehensive analysis
python generate_plots.py
```

### Using Individual Functions

```python
from evaluation.advanced_plots import AdvancedPlots

plotter = AdvancedPlots(figures_dir='my_figures')

# Generate specific plots
plotter.plot_poisoning_rate_performance()
plotter.plot_finetuning_analysis(epochs, asr_data, acc_data)
# ... etc
```

## Data Format Requirements

### For Poisoning Rate Analysis
```python
results_data = {
    'badnet': {
        'clean_accuracy': [90, 88, 85, 82, 79],  # for rates [0, 5, 10, 15, 20]
        'attack_success_rate': [0, 20, 40, 60, 80]
    },
    'blend': {
        'clean_accuracy': [88, 86, 83, 80, 77],
        'attack_success_rate': [0, 18, 35, 55, 75]
    }
    # ... more attacks
}
```

### For Feature Visualization
- Requires trained PyTorch models (nn.Module)
- Requires DataLoader objects for clean and infected samples
- Automatically extracts features from penultimate layer

### For Finetuning Analysis
```python
asr_data = {
    'defense_method': [initial_asr, asr_after_5_epochs, ..., final_asr]
}
acc_data = {
    'defense_method': [initial_acc, acc_after_5_epochs, ..., final_acc]
}
```

### For Training Loss Analysis
```python
training_history = {
    'method_name': {
        'clean_loss': [list of loss values per epoch],
        'backdoor_loss': [list of loss values per epoch],
        'val_acc': [list of accuracy values per epoch]  # optional
    }
}
```

## Installation

1. Install compatible dependencies:
```bash
pip install -r plotting_requirements.txt
```

2. If you encounter NumPy compatibility issues:
```bash
pip install numpy==1.24.3 matplotlib==3.7.2 --force-reinstall
```

## Integration with Your Framework

The plotting functions are designed to work seamlessly with your existing `run_experiments.py` pipeline:

1. **Modify `run_experiments.py`** to save training histories and intermediate results
2. **Use `generate_plots.py`** to automatically generate all plots after experiments
3. **Use `plot_examples.py`** to create custom visualizations with your existing results

## Customization

All plotting functions support:
- Custom color schemes
- Adjustable figure sizes
- Flexible save paths
- Different file formats (PNG, PDF, SVG)
- Custom titles and labels

Example customization:
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')  # Use seaborn style
plt.rcParams['figure.dpi'] = 300  # High DPI
plt.rcParams['font.size'] = 12   # Larger fonts
```

## Troubleshooting

1. **Import errors**: Install dependencies from `plotting_requirements.txt`
2. **Memory issues**: Reduce batch sizes for feature extraction
3. **No data**: Functions include synthetic data generation for demonstration
4. **NumPy compatibility**: Use numpy==1.24.3 for compatibility

## Output

All plots are saved as high-resolution PNG files (300 DPI) with:
- White backgrounds (publication ready)
- Tight bounding boxes
- Clear legends and labels
- Grid lines for easier reading

The generated plots are perfect for:
- Research papers
- Technical presentations
- Progress reports
- Experimental analysis