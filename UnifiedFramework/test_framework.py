"""
Test script for the unified framework
Validates all components work correctly
"""

import torch
import numpy as np
from pathlib import Path

print("Testing Unified Backdoor Attack and Defense Framework")
print("=" * 60)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from config import UnifiedConfig
    from attacks.backdoor_attacks import BadNetAttack, BlendAttack, SignalAttack, DynamicAttack, WaNetAttack
    from datasets.data_utils import get_dataset, get_dataloaders
    from models.architectures import create_model, WideResNet
    from evaluation.metrics import BackdoorMetrics
    from defenses.base_defense import StandardTrainer, BaseDefense
    from defenses.defense_methods import AIBDDefense, ABLDefense, CBDDefense, DBDDefense, NADDefense
    from evaluation.analyze_results import ResultsAnalyzer
    from utils.utilities import set_random_seed, count_parameters
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test 2: Configuration
print("\n[Test 2] Testing configuration...")
try:
    config = UnifiedConfig()
    args = config.get_args()
    print(f"✓ Configuration loaded")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Model: {args.model_arch}")
    print(f"  - Batch size: {args.batch_size}")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    exit(1)

# Test 3: Attacks
print("\n[Test 3] Testing attack methods...")
try:
    attacks = {
        'badnet': BadNetAttack(),
        'blend': BlendAttack(),
        'sig': SignalAttack(),
        'dynamic': DynamicAttack(),
        'wanet': WaNetAttack()
    }
    
    # Create dummy image
    dummy_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    for name, attack in attacks.items():
        poisoned = attack.apply_trigger(dummy_image)
        assert poisoned.shape == dummy_image.shape, f"Shape mismatch for {name}"
        assert poisoned.dtype == np.uint8, f"Type mismatch for {name}"
    
    print(f"✓ All {len(attacks)} attacks tested successfully")
except Exception as e:
    print(f"✗ Attack error: {e}")
    exit(1)

# Test 4: Model architectures
print("\n[Test 4] Testing model architectures...")
try:
    models_to_test = ['resnet18', 'wrn16-1', 'vgg16']
    
    for arch in models_to_test:
        model = create_model(arch, num_classes=10)
        assert isinstance(model, torch.nn.Module)
        
        # Count parameters
        params = count_parameters(model)
        print(f"  - {arch}: {params['trainable']:,} trainable params")
    
    print(f"✓ All {len(models_to_test)} architectures tested successfully")
except Exception as e:
    print(f"✗ Model error: {e}")
    exit(1)

# Test 5: Evaluation metrics
print("\n[Test 5] Testing evaluation metrics...")
try:
    metrics = BackdoorMetrics()
    
    # Test ACC
    predictions = np.array([0, 1, 2, 0, 1])
    labels = np.array([0, 1, 1, 0, 1])
    acc = metrics.calculate_acc(predictions, labels)
    assert 0 <= acc <= 100, "ACC out of range"
    
    # Test ASR
    asr = metrics.calculate_asr(predictions, 0)
    assert 0 <= asr <= 100, "ASR out of range"
    
    print(f"✓ Evaluation metrics tested successfully")
    print(f"  - ACC: {acc:.2f}%")
    print(f"  - ASR: {asr:.2f}%")
except Exception as e:
    print(f"✗ Metrics error: {e}")
    exit(1)

# Test 6: Defenses
print("\n[Test 6] Testing defense methods...")
try:
    model = create_model('resnet18', num_classes=10)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    defenses = {
        'base': BaseDefense(model, device),
        'standard': StandardTrainer(model, device),
    }
    
    print(f"✓ Basic defenses instantiated successfully")
except Exception as e:
    print(f"✗ Defense error: {e}")
    exit(1)

# Test 7: Utilities
print("\n[Test 7] Testing utilities...")
try:
    set_random_seed(42)
    
    # Create dummy model
    model = create_model('resnet18', num_classes=10)
    params = count_parameters(model)
    
    print(f"✓ Utilities tested successfully")
    print(f"  - Total parameters: {params['total']:,}")
    print(f"  - Trainable parameters: {params['trainable']:,}")
except Exception as e:
    print(f"✗ Utility error: {e}")
    exit(1)

# Test 8: Directory structure
print("\n[Test 8] Checking directory structure...")
try:
    required_dirs = [
        'attacks',
        'defenses',
        'datasets',
        'models',
        'evaluation',
        'utils',
        'results'
    ]
    
    missing = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(dir_name)
    
    if missing:
        print(f"✗ Missing directories: {missing}")
        exit(1)
    
    print(f"✓ All required directories present")
except Exception as e:
    print(f"✗ Directory check error: {e}")
    exit(1)

# Final summary
print("\n" + "=" * 60)
print("All tests passed successfully!")
print("=" * 60)
print("\nFramework is ready to use. Run experiments with:")
print("  python run_experiments.py --dataset CIFAR10 --epochs 100")
print("\nOr try the quick start examples:")
print("  python quick_start.py --example 1")
