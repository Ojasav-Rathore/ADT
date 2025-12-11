"""
Unified Configuration for Backdoor Attack and Defense Framework
Supports: CIFAR-10, GTSRB datasets
Attacks: BadNet, Blend, Dynamic, SIG, WaNet
Defenses: AIBD, ABL, CBD, DBD, NAD
"""

import argparse
from typing import List


class UnifiedConfig:
    """Unified configuration for all experiments"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Unified Backdoor Attack and Defense Framework')
        self._add_dataset_args()
        self._add_attack_args()
        self._add_defense_args()
        self._add_model_args()
        self._add_training_args()
        self._add_evaluation_args()
        
    def _add_dataset_args(self):
        """Dataset configuration arguments"""
        group = self.parser.add_argument_group('Dataset')
        group.add_argument('--dataset', type=str, default='CIFAR10', 
                          choices=['CIFAR10', 'GTSRB'],
                          help='Dataset to use')
        group.add_argument('--data_path', type=str, default='./data',
                          help='Path to dataset')
        group.add_argument('--num_classes', type=int, default=10,
                          help='Number of classes (10 for CIFAR10, 43 for GTSRB)')
        group.add_argument('--img_height', type=int, default=32,
                          help='Image height')
        group.add_argument('--img_width', type=int, default=32,
                          help='Image width')
        group.add_argument('--num_workers', type=int, default=4,
                          help='Number of data loading workers')
    
    def _add_attack_args(self):
        """Backdoor attack configuration arguments"""
        group = self.parser.add_argument_group('Attack')
        group.add_argument('--attack_method', type=str, default='badnet',
                          choices=['badnet', 'blend', 'dynamic', 'sig', 'wanet'],
                          help='Backdoor attack method')
        group.add_argument('--poison_rate', type=float, default=0.1,
                          help='Poisoning rate for training data')
        group.add_argument('--target_label', type=int, default=0,
                          help='Target label for backdoor attack')
        group.add_argument('--target_type', type=str, default='all2one',
                          choices=['all2one', 'all2all', 'clean_label'],
                          help='Type of backdoor attack')
        group.add_argument('--trigger_width', type=int, default=4,
                          help='Width of trigger pattern')
        group.add_argument('--trigger_height', type=int, default=4,
                          help='Height of trigger pattern')
        group.add_argument('--trigger_alpha', type=float, default=0.2,
                          help='Alpha value for blending triggers')
    
    def _add_defense_args(self):
        """Defense method configuration arguments"""
        group = self.parser.add_argument_group('Defense')
        group.add_argument('--defense_method', type=str, default='none',
                          choices=['none', 'aibd', 'abl', 'cbd', 'dbd', 'nad'],
                          help='Defense method to use')
        
        # AIBD specific
        group.add_argument('--aibd_adv_eps', type=float, default=1.0,
                          help='AIBD adversarial epsilon')
        group.add_argument('--aibd_adv_alpha', type=float, default=1/255,
                          help='AIBD adversarial alpha')
        group.add_argument('--aibd_adv_steps', type=int, default=255,
                          help='AIBD adversarial attack steps')
        group.add_argument('--aibd_iso_ratio', type=list, 
                          default=[0.20, 0.15, 0.10, 0.05],
                          help='AIBD isolation ratios')
        
        # ABL specific
        group.add_argument('--abl_isolation_ratio', type=float, default=0.01,
                          help='ABL isolation ratio')
        group.add_argument('--abl_tuning_epochs', type=int, default=10,
                          help='ABL tuning epochs')
        group.add_argument('--abl_unlearning_epochs', type=int, default=5,
                          help='ABL unlearning epochs')
        group.add_argument('--abl_flooding', type=float, default=0.5,
                          help='ABL flooding value')
        
        # CBD specific
        group.add_argument('--cbd_epochs', type=int, default=100,
                          help='CBD training epochs')
        group.add_argument('--cbd_lambda', type=float, default=1.0,
                          help='CBD lambda for mutual information')
        
        # DBD specific
        group.add_argument('--dbd_epochs', type=int, default=100,
                          help='DBD training epochs')
        group.add_argument('--dbd_n_classes', type=int, default=10,
                          help='DBD number of classes')
        
        # NAD specific
        group.add_argument('--nad_epochs', type=int, default=20,
                          help='NAD training epochs')
        group.add_argument('--nad_ratio', type=float, default=0.05,
                          help='NAD ratio of training data to use')
        group.add_argument('--nad_beta1', type=int, default=500,
                          help='NAD beta for low layer')
        group.add_argument('--nad_beta2', type=int, default=1000,
                          help='NAD beta for middle layer')
        group.add_argument('--nad_beta3', type=int, default=1000,
                          help='NAD beta for high layer')
    
    def _add_model_args(self):
        """Model configuration arguments"""
        group = self.parser.add_argument_group('Model')
        group.add_argument('--model_arch', type=str, default='resnet18',
                          choices=['resnet18', 'resnet34', 'wrn16-1', 'vgg16'],
                          help='Model architecture')
        group.add_argument('--pretrained', action='store_true',
                          help='Use pretrained model')
    
    def _add_training_args(self):
        """Training configuration arguments"""
        group = self.parser.add_argument_group('Training')
        group.add_argument('--batch_size', type=int, default=128,
                          help='Batch size')
        group.add_argument('--epochs', type=int, default=100,
                          help='Number of epochs')
        group.add_argument('--lr', type=float, default=0.1,
                          help='Learning rate')
        group.add_argument('--lr_scheduler', type=str, default='cosine',
                          choices=['step', 'cosine', 'exponential'],
                          help='Learning rate scheduler')
        group.add_argument('--momentum', type=float, default=0.9,
                          help='Momentum for SGD')
        group.add_argument('--weight_decay', type=float, default=1e-4,
                          help='Weight decay')
        group.add_argument('--device', type=str, default='cuda:0',
                          help='Device to use')
        group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    
    def _add_evaluation_args(self):
        """Evaluation configuration arguments"""
        group = self.parser.add_argument_group('Evaluation')
        group.add_argument('--save_results', action='store_true',
                          help='Save evaluation results')
        group.add_argument('--results_dir', type=str, default='./results',
                          help='Directory to save results')
        group.add_argument('--log_dir', type=str, default='./logs',
                          help='Directory to save logs')
        group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                          help='Directory to save checkpoints')
        group.add_argument('--print_freq', type=int, default=100,
                          help='Frequency of printing')
    
    def get_args(self):
        """Parse and return arguments"""
        return self.parser.parse_args()
    
    def get_parser(self):
        """Return the argument parser"""
        return self.parser


def get_default_config():
    """Get default configuration"""
    config = UnifiedConfig()
    return config.get_args()


def get_config_for_dataset(dataset: str) -> dict:
    """Get default config values for a specific dataset"""
    configs = {
        'CIFAR10': {
            'num_classes': 10,
            'img_height': 32,
            'img_width': 32,
        },
        'GTSRB': {
            'num_classes': 43,
            'img_height': 32,
            'img_width': 32,
        }
    }
    return configs.get(dataset, configs['CIFAR10'])


def get_config_for_attack(attack: str) -> dict:
    """Get default config values for a specific attack"""
    configs = {
        'badnet': {
            'trigger_width': 4,
            'trigger_height': 4,
            'trigger_alpha': 1.0,
        },
        'blend': {
            'trigger_width': 32,
            'trigger_height': 32,
            'trigger_alpha': 0.2,
        },
        'dynamic': {
            'trigger_width': 32,
            'trigger_height': 32,
            'trigger_alpha': 1.0,
        },
        'sig': {
            'trigger_width': 32,
            'trigger_height': 32,
            'trigger_alpha': 0.2,
        },
        'wanet': {
            'trigger_width': 32,
            'trigger_height': 32,
            'trigger_alpha': 1.0,
        }
    }
    return configs.get(attack, configs['badnet'])


if __name__ == '__main__':
    config = UnifiedConfig()
    args = config.get_args()
    print(args)
