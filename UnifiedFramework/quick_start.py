"""
Quick start examples for the unified framework
"""

import torch
from config import UnifiedConfig
from run_experiments import ExperimentPipeline


def example_1_single_experiment():
    """
    Example 1: Run a single experiment with BadNet attack and ABL defense
    """
    print("="*60)
    print("Example 1: Single Experiment (BadNet + ABL on CIFAR-10)")
    print("="*60)
    
    config = UnifiedConfig()
    args = config.get_args()
    
    # Configure for quick test
    args.dataset = 'CIFAR10'
    args.epochs = 5  # Reduced for quick test
    args.batch_size = 128
    args.poison_rate = 0.1
    args.target_label = 0
    args.model_arch = 'resnet18'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    pipeline = ExperimentPipeline(args)
    results = pipeline.run_single_experiment('badnet', 'abl')
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


def example_2_all_attacks():
    """
    Example 2: Test all attacks with one defense method
    """
    print("\n" + "="*60)
    print("Example 2: All Attacks with ABL Defense")
    print("="*60)
    
    config = UnifiedConfig()
    args = config.get_args()
    
    args.dataset = 'CIFAR10'
    args.epochs = 3  # Reduced for quick test
    args.batch_size = 128
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    pipeline = ExperimentPipeline(args)
    
    attacks = ['badnet', 'blend', 'sig']  # Subset for quick test
    defenses = ['none', 'abl']
    
    results = pipeline.run_all_experiments(attacks, defenses)
    pipeline.save_results(results, 'example_results.json')


def example_3_gtsrb():
    """
    Example 3: Test on GTSRB dataset
    """
    print("\n" + "="*60)
    print("Example 3: GTSRB Dataset with AIBD Defense")
    print("="*60)
    
    config = UnifiedConfig()
    args = config.get_args()
    
    args.dataset = 'GTSRB'
    args.num_classes = 43
    args.epochs = 3
    args.batch_size = 128
    args.poison_rate = 0.05
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    pipeline = ExperimentPipeline(args)
    results = pipeline.run_single_experiment('badnet', 'aibd')
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


def example_4_custom_config():
    """
    Example 4: Using custom configuration
    """
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    config = UnifiedConfig()
    args = config.get_args()
    
    # Custom parameters
    args.dataset = 'CIFAR10'
    args.model_arch = 'wrn16-1'
    args.attack_method = 'blend'
    args.defense_method = 'cbd'
    args.poison_rate = 0.15
    args.target_label = 2
    args.epochs = 5
    args.lr = 0.01
    args.batch_size = 64
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    pipeline = ExperimentPipeline(args)
    results = pipeline.run_single_experiment(args.attack_method, args.defense_method)
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start examples')
    parser.add_argument('--example', type=int, default=1, 
                       choices=[1, 2, 3, 4],
                       help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_1_single_experiment()
    elif args.example == 2:
        example_2_all_attacks()
    elif args.example == 3:
        example_3_gtsrb()
    elif args.example == 4:
        example_4_custom_config()
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
