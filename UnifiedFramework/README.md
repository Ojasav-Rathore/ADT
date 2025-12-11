# Unified Backdoor Attack and Defense Framework

A comprehensive framework for evaluating backdoor attacks and defenses on deep neural networks. This framework implements multiple state-of-the-art defense mechanisms against backdoor attacks on CIFAR-10 and GTSRB datasets.

## Overview

This unified framework provides:

- **5 Attack Methods**: BadNet, Blend, Dynamic, SIG (Signal), WaNet
- **5 Defense Methods**: AIBD, ABL, CBD, DBD, NAD
- **2 Datasets**: CIFAR-10 (10 classes), GTSRB (43 classes)
- **Comprehensive Evaluation**: ACC (Clean Accuracy), ASR (Attack Success Rate), ROC curves

## Acronyms

### Attacks
- **BadNet**: Square trigger pattern in corner
- **Blend**: Transparent blended trigger
- **Dynamic**: Parameterized adaptive trigger
- **SIG**: Frequency-based signal trigger
- **WaNet**: Geometric warping-based trigger

### Defenses
- **AIBD**: Adversarial-Inspired Backdoor Defense
- **ABL**: Anti-Backdoor Learning
- **CBD**: Causality-inspired Backdoor Defense
- **DBD**: Backdoor Defense via Decoupling
- **NAD**: Neural Attention Distillation

## Directory Structure

```
UnifiedFramework/
├── attacks/
│   ├── backdoor_attacks.py          # Attack implementations
│   └── __init__.py
├── defenses/
│   ├── base_defense.py              # Base defense class
│   ├── defense_methods.py           # Specific defense implementations
│   └── __init__.py
├── datasets/
│   ├── data_utils.py                # Dataset loading utilities
│   └── __init__.py
├── models/
│   ├── architectures.py             # Model architectures (ResNet, WideResNet, VGG)
│   └── __init__.py
├── evaluation/
│   ├── metrics.py                   # Evaluation metrics (ACC, ASR, ROC)
│   ├── analyze_results.py           # Results analysis and visualization
│   └── __init__.py
├── config.py                        # Unified configuration
├── run_experiments.py               # Main experimental pipeline
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
cd UnifiedFramework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
```bash
# CIFAR-10 will be downloaded automatically
# For GTSRB, download from: https://benchmark.ini.rub.de/
# Extract to: ./data/GTSRB/
```

## Quick Start

### Run All Experiments

```bash
python run_experiments.py --dataset CIFAR10 --epochs 100 --batch_size 128
```

### Run Specific Attack and Defense

```bash
python run_experiments.py \
    --dataset CIFAR10 \
    --attack_method badnet \
    --defense_method abl \
    --epochs 100 \
    --batch_size 128
```

### Custom Configuration

```bash
python run_experiments.py \
    --dataset GTSRB \
    --model_arch resnet18 \
    --poison_rate 0.1 \
    --target_label 0 \
    --epochs 100 \
    --lr 0.1 \
    --device cuda:0
```

## Configuration Options

### Dataset Options
```
--dataset CIFAR10 | GTSRB
--data_path ./data
--num_workers 4
```

### Attack Options
```
--attack_method badnet | blend | dynamic | sig | wanet
--poison_rate 0.1          # Poisoning rate
--target_label 0           # Target class
--trigger_width 4          # Trigger width
--trigger_height 4         # Trigger height
--trigger_alpha 0.2        # Blend transparency
```

### Defense Options
```
--defense_method none | aibd | abl | cbd | dbd | nad

# AIBD specific
--aibd_adv_eps 1.0
--aibd_adv_alpha 0.00392
--aibd_adv_steps 255

# ABL specific
--abl_isolation_ratio 0.01
--abl_tuning_epochs 10
--abl_unlearning_epochs 5

# NAD specific
--nad_epochs 20
--nad_ratio 0.05
```

### Model Options
```
--model_arch resnet18 | resnet34 | wrn16-1 | vgg16
--pretrained False
```

### Training Options
```
--epochs 100
--batch_size 128
--lr 0.1
--momentum 0.9
--weight_decay 1e-4
--device cuda:0 | cpu
```

## Usage Examples

### Example 1: Evaluate BadNet Attack with ABL Defense on CIFAR-10

```python
from config import UnifiedConfig
from run_experiments import ExperimentPipeline

config = UnifiedConfig()
args = config.get_args()
args.dataset = 'CIFAR10'
args.attack_method = 'badnet'
args.defense_method = 'abl'
args.epochs = 100

pipeline = ExperimentPipeline(args)
results = pipeline.run_single_experiment('badnet', 'abl')
```

### Example 2: Run All Attacks Against One Defense

```python
pipeline = ExperimentPipeline(args)
attacks = ['badnet', 'blend', 'sig', 'dynamic', 'wanet']
results = pipeline.run_all_experiments(attacks, ['abl'])
```

### Example 3: Analyze Results

```bash
python evaluation/analyze_results.py results_20240101_120000.json ./report_output
```

This generates:
- `acc_comparison.png` - Clean accuracy comparison
- `asr_comparison.png` - Attack success rate comparison
- `defense_effectiveness.png` - Defense effectiveness metrics
- `acc_heatmap.png` - ACC across attacks and defenses
- `asr_heatmap.png` - ASR across attacks and defenses

## Results Format

Results are saved as JSON with the following structure:

```json
{
  "attack": "badnet",
  "defense": "abl",
  "dataset": "CIFAR10",
  "ACC_before_defense": 85.5,
  "ASR_before_defense": 99.8,
  "ACC_after_defense": 86.2,
  "ASR_after_defense": 3.2,
  "ACC_improvement": 0.7,
  "ASR_reduction": 96.6
}
```

## Evaluation Metrics

### 1. Clean Accuracy (ACC)
Accuracy on clean, non-poisoned test samples.
- **Range**: 0-100%
- **Goal**: Maintain high ACC after defense

### 2. Attack Success Rate (ASR)
Percentage of poisoned samples classified as target label.
- **Range**: 0-100%
- **Goal**: Reduce ASR to near 0% after defense

### 3. ROC Curve (TPR vs FPR)
Trade-off between true positive rate and false positive rate for backdoor detection.
- **Area Under Curve (AUC)**: 0-1
- **Goal**: Maximize AUC (close to 1.0)

## Performance Benchmarks

Expected results on CIFAR-10 (baseline - no defense):

| Attack | ACC (%) | ASR (%) |
|--------|---------|---------|
| BadNet | 85.2 | 99.9 |
| Blend | 84.8 | 98.5 |
| SIG | 85.1 | 97.2 |
| Dynamic | 84.5 | 99.1 |
| WaNet | 83.9 | 96.8 |

## Key Features

### 1. Modular Architecture
- Easy to add new attacks
- Easy to implement new defenses
- Flexible configuration system

### 2. Comprehensive Evaluation
- Clean accuracy measurement
- Attack success rate calculation
- ROC curve analysis
- Comparison across all combinations

### 3. Reproducibility
- Fixed random seeds
- Configuration saving
- Detailed logging

### 4. Visualization Tools
- Comparison plots
- Heatmaps
- ROC curves
- Summary statistics

## Extending the Framework

### Adding a New Attack

```python
# In attacks/backdoor_attacks.py

class YourNewAttack:
    def __init__(self, **kwargs):
        # Initialize attack parameters
        pass
    
    def apply_trigger(self, image: np.ndarray) -> np.ndarray:
        # Implement trigger application
        return triggered_image

# Register in BackdoorAttackFactory
attacks = {
    'your_attack': YourNewAttack,
    ...
}
```

### Adding a New Defense

```python
# In defenses/defense_methods.py

class YourDefense(BaseDefense):
    def __init__(self, model, device='cuda:0'):
        super().__init__(model, device)
    
    def defend(self, train_loader, val_loader, epochs):
        # Implement defense procedure
        pass

# Register in get_defense_method
defenses = {
    'your_defense': YourDefense,
    ...
}
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{unified_backdoor_framework,
  title={Unified Backdoor Attack and Defense Framework},
  author={Your Name},
  year={2024}
}
```

## References

1. **AIBD**: "Adversarial-Inspired Backdoor Defense via Bridging Backdoor and Adversarial Attacks" (AAAI 2025)
2. **ABL**: "Anti-Backdoor Learning: Training Clean Models on Poisoned Data" (NeurIPS 2021)
3. **CBD**: "Backdoor Defense via Deconfounded Representation Learning" (CVPR 2023)
4. **DBD**: "Backdoor Defense via Decoupling the Training Process" (ICLR 2023)
5. **NAD**: "Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks" (ICLR 2021)

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Changelog

### Version 1.0 (Initial Release)
- Implemented 5 attack methods
- Implemented 5 defense methods
- Support for CIFAR-10 and GTSRB
- Comprehensive evaluation metrics
- Visualization tools
- Complete documentation
