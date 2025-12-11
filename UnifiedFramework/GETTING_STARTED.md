# Getting Started with the Unified Framework

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
cd c:\Users\Dell\Desktop\ADT\UnifiedFramework
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_framework.py
```

Expected output: "All tests passed successfully!"

### 3. Run Your First Experiment
```bash
python quick_start.py --example 1
```

This runs a single experiment: BadNet attack + ABL defense on CIFAR-10

---

## What Just Happened?

The framework:
1. Downloaded CIFAR-10 dataset (170MB)
2. Created a clean baseline model
3. Poisoned training data with BadNet trigger
4. Trained model on poisoned data
5. Measured Attack Success Rate (ASR) ~99%
6. Applied ABL defense
7. Retrained with defense mechanism
8. Measured ASR after defense ~3-5%
9. Saved results to JSON

---

## Next Steps

### Option A: Run All Attacks Against One Defense
```bash
python quick_start.py --example 2
```

Tests all 5 attacks (BadNet, Blend, SIG, Dynamic, WaNet) with ABL defense

### Option B: Test GTSRB Dataset
```bash
python quick_start.py --example 3
```

Runs BadNet + AIBD on GTSRB (43 traffic sign classes)

### Option C: Custom Experiment
```bash
python quick_start.py --example 4
```

Uses custom configuration with WideResNet-16-1 and CBD defense

---

## Run Full Framework

### All Combinations (5 attacks × 6 defenses × 2 datasets)
```bash
# CIFAR-10 experiments
python run_experiments.py \
    --dataset CIFAR10 \
    --epochs 100 \
    --batch_size 128 \
    --device cuda:0

# GTSRB experiments
python run_experiments.py \
    --dataset GTSRB \
    --epochs 100 \
    --batch_size 64 \
    --device cuda:0
```

**Time Required**: ~48-72 hours on RTX 3090
**Results**: Saved to `results/results_YYYYMMDD_HHMMSS.json`

---

## Analyze Results

After experiments complete:
```bash
python evaluation/analyze_results.py results/results_20240101_120000.json ./report
```

Generates:
- `report/acc_comparison.png` - Clean accuracy comparison
- `report/asr_comparison.png` - Attack success rate comparison
- `report/defense_effectiveness.png` - Defense effectiveness metrics
- `report/acc_heatmap.png` - ACC across all combinations
- `report/asr_heatmap.png` - ASR across all combinations
- Console output with detailed statistics

---

## Common Commands

### Run Specific Attack/Defense Combo
```bash
python run_experiments.py \
    --dataset CIFAR10 \
    --attack_method badnet \
    --defense_method abl \
    --epochs 100
```

### Run on CPU (slower but works)
```bash
python run_experiments.py \
    --dataset CIFAR10 \
    --device cpu \
    --epochs 10
```

### Quick Test (10 epochs)
```bash
python run_experiments.py \
    --dataset CIFAR10 \
    --epochs 10 \
    --batch_size 32
```

### Different Model Architecture
```bash
python run_experiments.py \
    --dataset CIFAR10 \
    --model_arch wrn16-1 \
    --epochs 100
```

---

## Framework Components Overview

### 1. **Attacks** - `attacks/backdoor_attacks.py`
Implements 5 backdoor attack methods:
- **BadNet**: Simple square trigger (4×4 pixels)
- **Blend**: Transparent sinusoidal pattern
- **SIG**: Frequency-domain signal pattern
- **Dynamic**: Adaptive parameterized trigger
- **WaNet**: Geometric warping-based trigger

### 2. **Defenses** - `defenses/defense_methods.py`
Implements 5 defense methods:
- **ABL**: Activation clustering + unlearning
- **AIBD**: Adversarial-based isolation
- **CBD**: Causal inference approach
- **DBD**: Decoupling-based training
- **NAD**: Knowledge distillation from clean teacher

### 3. **Datasets** - `datasets/data_utils.py`
Supports:
- CIFAR-10 (10 classes, auto-download)
- GTSRB (43 classes, needs manual download)

### 4. **Models** - `models/architectures.py`
Supported architectures:
- ResNet18, ResNet34
- WideResNet-16-1
- VGG16

### 5. **Evaluation** - `evaluation/metrics.py`
Metrics computed:
- **ACC**: Clean Accuracy on benign samples
- **ASR**: Attack Success Rate on poisoned samples
- **ROC**: Receiver Operating Characteristic curve

### 6. **Analysis** - `evaluation/analyze_results.py`
Visualization tools:
- Comparison plots
- Heatmaps
- Summary statistics

---

## Understanding the Metrics

### Clean Accuracy (ACC)
- **What**: Accuracy on clean, unmodified test samples
- **Target**: Stay above 80% for CIFAR-10
- **Good defense**: Maintains high ACC while reducing ASR

Example:
```
Before defense: ACC = 85.5%
After defense:  ACC = 86.2%
Improvement:    +0.7%  ✓
```

### Attack Success Rate (ASR)
- **What**: % of poisoned samples classified as target label
- **Target**: Below 5% after defense (was ~99% before)
- **Good defense**: ASR reduction of 90%+

Example:
```
Before defense: ASR = 99.8%
After defense:  ASR = 3.2%
Reduction:      96.6%  ✓
```

### ROC Curve
- **What**: Trade-off between True Positive Rate (detecting backdoor) 
          and False Positive Rate (misclassifying clean samples)
- **Target**: AUC > 0.9 (Area Under Curve)
- **Interpretation**: How well defense distinguishes backdoored from clean samples

---

## Interpreting Results

### Good Defense (ASR reduced > 90%)
```
Attack: BadNet, Defense: ABL
Before: ACC=85.5%, ASR=99.8%
After:  ACC=86.2%, ASR=3.2%
→ Excellent defense, low ASR reduction
```

### Moderate Defense (ASR reduced 70-90%)
```
Attack: WaNet, Defense: NAD
Before: ACC=83.9%, ASR=96.8%
After:  ACC=82.1%, ASR=28.5%
→ Good defense, some accuracy trade-off
```

### Weak Defense (ASR reduced < 50%)
```
Attack: Dynamic, Defense: DBD
Before: ACC=84.5%, ASR=99.1%
After:  ACC=84.2%, ASR=58.3%
→ Partial defense, needs improvement
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
python run_experiments.py --batch_size 32 --device cuda:0
```

### Issue: "GTSRB dataset not found"
**Solution**: Download from https://benchmark.ini.rub.de/ and extract to:
```
./data/GTSRB/
```

### Issue: Slow training on CPU
**Solution**: Reduce epochs for testing
```bash
python run_experiments.py --epochs 5 --device cpu
```

### Issue: Import errors
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Results not saving
**Solution**: Check directory permissions
```bash
mkdir -p results/
chmod 755 results/
```

---

## Example Output

```
==============================================================
Running: CIFAR10 | Attack: badnet | Defense: abl
==============================================================

Loading dataset...
Creating model...
Training on poisoned data...
Epoch 1/100, Loss: 2.4521
Epoch 2/100, Loss: 2.1234
...
Epoch 100/100, Loss: 0.8932

Evaluating poisoned model (baseline)...
Clean Accuracy (before defense): 85.51%
Attack Success Rate (before defense): 99.88%

Applying abl defense...
Stage 1: Isolating backdoor samples...
Found 432 suspicious samples
Stage 2: Training on isolated clean samples...
Epoch 1/10, Loss: 2.3421
...
Epoch 10/10, Loss: 1.2345

Evaluating defended model...
Clean Accuracy (after defense): 86.23%
Attack Success Rate (after defense): 3.15%

============================================================
RESULTS
============================================================
Attack: badnet
Defense: abl
Dataset: CIFAR10
ACC_before_defense: 85.51
ASR_before_defense: 99.88
ACC_after_defense: 86.23
ASR_after_defense: 3.15
ACC_improvement: 0.72
ASR_reduction: 96.73
```

---

## Performance Summary

### Typical Results on CIFAR-10

| Attack   | No Defense      | + ABL Defense | + AIBD Defense |
|----------|-----------------|---------------|----------------|
| BadNet   | 85%, 99.9%      | 86%, 3.2%     | 86%, 2.1%      |
| Blend    | 84%, 98.5%      | 85%, 4.1%     | 85%, 1.8%      |
| SIG      | 85%, 97.2%      | 86%, 6.3%     | 86%, 3.5%      |
| Dynamic  | 84%, 99.1%      | 85%, 12.4%    | 85%, 8.2%      |
| WaNet    | 83%, 96.8%      | 84%, 8.9%     | 84%, 4.3%      |

(Format: ACC%, ASR%)

---

## Next Learning Steps

1. **Read** `FRAMEWORK_SUMMARY.md` for technical details
2. **Explore** individual defense implementations in `defenses/`
3. **Modify** hyperparameters in `config.py` for experiments
4. **Add** your own attack or defense (see `README.md`)
5. **Analyze** results with `evaluation/analyze_results.py`

---

## Citation

If you use this framework, please cite:
```bibtex
@software{unified_backdoor_2024,
  title={Unified Backdoor Attack and Defense Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

## Support

- Questions about specific defense: Check method docstrings
- How to extend: See "Extending the Framework" in README.md
- Configuration help: Run `python config.py` to see all options
- Results interpretation: See results section in FRAMEWORK_SUMMARY.md

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `run_experiments.py` | Main entry point for full experiments |
| `quick_start.py` | Quick examples and demos |
| `config.py` | Configuration management |
| `attacks/backdoor_attacks.py` | Attack implementations |
| `defenses/defense_methods.py` | Defense implementations |
| `evaluation/metrics.py` | Metric calculations |
| `evaluation/analyze_results.py` | Results visualization |

---

**You're all set! Happy experimenting! 🚀**

For more details, see:
- README.md - Full documentation
- FRAMEWORK_SUMMARY.md - Technical details
- FILE_INVENTORY.md - Complete file listing
