"""
Main experimental pipeline for unified backdoor attack and defense framework
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Import modules
from config import UnifiedConfig, get_config_for_dataset, get_config_for_attack
from attacks.backdoor_attacks import (
    BackdoorAttackFactory, apply_backdoor_to_dataset
)
from datasets.data_utils import (
    get_dataloaders, get_dataset, get_data_transforms, 
    PoisonedDataset, create_poisoned_loader
)
from models.architectures import create_model
from evaluation.metrics import BackdoorMetrics, plot_roc_curve, plot_comparison_metrics
from defenses.defense_methods import get_defense_method
from defenses.base_defense import StandardTrainer


class ExperimentPipeline:
    """Unified experimental pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.results = {}
        self.models = {}
        
        # Create directories
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Update dataset-specific config
        dataset_config = get_config_for_dataset(args.dataset)
        for key, val in dataset_config.items():
            setattr(args, key, val)
    
    def run_single_experiment(self, attack_method: str, defense_method: str) -> Dict:
        """
        Run single experiment with specified attack and defense
        
        Args:
            attack_method: Attack to use
            defense_method: Defense to use
        
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*60}")
        print(f"Running: {self.args.dataset} | Attack: {attack_method} | Defense: {defense_method}")
        print(f"{'='*60}\n")
        
        # Set random seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # Get dataset
        print("Loading dataset...")
        train_loader, test_loader = get_dataloaders(
            self.args.dataset, 
            self.args.data_path,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            augment=True
        )
        
        # Load base dataset WITHOUT transforms for poisoning
        base_train_dataset = get_dataset(self.args.dataset, self.args.data_path, 
                                        train=True, transform=None)
        base_test_dataset = get_dataset(self.args.dataset, self.args.data_path,
                                       train=False, transform=None)
        
        # Create model
        print("Creating model...")
        model = create_model(self.args.model_arch, self.args.num_classes)
        model = model.to(self.device)
        
        # Step 1: Train on poisoned data (baseline - no defense)
        print(f"\nStep 1: Training on poisoned data with {attack_method} attack...")
        poisoned_train_dataset, poison_indices = self._create_poisoned_dataset(
            base_train_dataset, attack_method
        )
        
        poisoned_train_loader = create_poisoned_loader(
            self.args.dataset, self.args.data_path, poisoned_train_dataset,
            batch_size=self.args.batch_size
        )
        
        # Train poisoned model
        trainer = StandardTrainer(model, self.args.device)
        experiment_name = f"{attack_method}_{defense_method}_poisoned"
        trainer.train_model(poisoned_train_loader, test_loader, 
                           self.args.epochs, self.args.lr, verbose=True,
                           save_losses=True, experiment_name=experiment_name)
        
        # Evaluate poisoned model (baseline)
        print("\nEvaluating poisoned model (baseline)...")
        metrics = BackdoorMetrics()
        
        # Test on clean data - with transforms
        _, test_transform = get_data_transforms(self.args.dataset, augment=False)
        clean_test_dataset = get_dataset(self.args.dataset, self.args.data_path,
                                        train=False, transform=test_transform)
        test_loader_clean = torch.utils.data.DataLoader(
            clean_test_dataset, batch_size=self.args.batch_size, shuffle=False
        )
        
        clean_acc, clean_preds, clean_labels = trainer.evaluate(test_loader_clean)
        print(f"Clean Accuracy (before defense): {clean_acc*100:.2f}%")
        
        # Test on poisoned samples (handle clean training case)
        if attack_method.lower() == 'none':
            # No attack means ASR should be 0
            asr_before = 0.0
            print(f"Attack Success Rate (before defense): {asr_before:.2f}% (no attack)")
        else:
            poisoned_test_dataset, poison_test_indices = self._create_poisoned_dataset(
                base_test_dataset, attack_method
            )
            poisoned_test_loader = create_poisoned_loader(
                self.args.dataset, self.args.data_path, poisoned_test_dataset,
                batch_size=self.args.batch_size
            )
            
            _, poison_preds, poison_labels = trainer.evaluate(poisoned_test_loader)
            asr_before = np.mean(poison_preds == self.args.target_label) * 100
            print(f"Attack Success Rate (before defense): {asr_before:.2f}%")
        
        results = {
            'attack': attack_method,
            'defense': defense_method,
            'dataset': self.args.dataset,
            'ACC_before_defense': clean_acc * 100,
            'ASR_before_defense': asr_before,
        }
        
        # Step 2: Apply defense
        if defense_method.lower() != 'none':
            print(f"\nStep 2: Applying {defense_method} defense...")
            
            # Create new poisoned model for defense
            model_defended = create_model(self.args.model_arch, self.args.num_classes)
            model_defended = model_defended.to(self.device)
            
            # Retrain with defense
            if defense_method.lower() == 'abl':
                from defenses.defense_methods import ABLDefense
                defense = ABLDefense(model_defended, self.args.device)
            elif defense_method.lower() == 'aibd':
                from defenses.defense_methods import AIBDDefense
                defense = AIBDDefense(model_defended, self.args.device)
            elif defense_method.lower() == 'cbd':
                from defenses.defense_methods import CBDDefense
                defense = CBDDefense(model_defended, self.args.device)
            elif defense_method.lower() == 'dbd':
                from defenses.defense_methods import DBDDefense
                defense = DBDDefense(model_defended, self.args.device)
            elif defense_method.lower() == 'nad':
                from defenses.defense_methods import NADDefense
                # Train teacher model on clean data
                teacher = create_model(self.args.model_arch, self.args.num_classes)
                teacher_trainer = StandardTrainer(teacher, self.args.device)
                teacher_experiment_name = f"{attack_method}_{defense_method}_teacher"
                teacher_trainer.train_model(test_loader_clean, test_loader_clean, 
                                           self.args.epochs, self.args.lr, verbose=False,
                                           save_losses=True, experiment_name=teacher_experiment_name)
                defense = NADDefense(model_defended, teacher, self.args.device)
            else:
                defense = StandardTrainer(model_defended, self.args.device)
            
            # Apply defense
            defense_experiment_name = f"{attack_method}_{defense_method}_defended"
            if defense_method.lower() == 'nad':
                defense.defend(poisoned_train_loader, test_loader_clean, 
                             test_loader_clean, self.args.epochs, experiment_name=defense_experiment_name)
            else:
                defense.defend(poisoned_train_loader, test_loader_clean, 
                             self.args.epochs, experiment_name=defense_experiment_name)
            
            # Evaluate defended model
            print("\nEvaluating defended model...")
            clean_acc_after, _, _ = defense.evaluate(test_loader_clean)
            print(f"Clean Accuracy (after defense): {clean_acc_after*100:.2f}%")
            
            # Handle ASR evaluation for defended model
            if attack_method.lower() == 'none':
                asr_after = 0.0
                print(f"Attack Success Rate (after defense): {asr_after:.2f}% (no attack)")
            else:
                # Recreate poisoned test loader for consistency 
                poisoned_test_dataset, _ = self._create_poisoned_dataset(
                    base_test_dataset, attack_method
                )
                poisoned_test_loader = create_poisoned_loader(
                    self.args.dataset, self.args.data_path, poisoned_test_dataset,
                    batch_size=self.args.batch_size
                )
                asr_after_preds, _, _ = defense.evaluate(poisoned_test_loader)
                asr_after = np.mean(asr_after_preds == self.args.target_label) * 100
                print(f"Attack Success Rate (after defense): {asr_after:.2f}%")
            
            results['ACC_after_defense'] = clean_acc_after * 100
            results['ASR_after_defense'] = asr_after
            results['ACC_improvement'] = clean_acc_after * 100 - clean_acc * 100
            results['ASR_reduction'] = asr_before - asr_after
        
        return results
    
    def _create_poisoned_dataset(self, base_dataset, attack_method: str) -> Tuple[list, list]:
        """Create poisoned version of dataset"""
        # Handle clean training (no attack)
        if attack_method.lower() == 'none':
            print("Clean training - no attack applied")
            return list(base_dataset), []
        
        # Update attack-specific config
        attack_config = get_config_for_attack(attack_method)
        
        # Create attack
        attack = BackdoorAttackFactory.create_attack(
            attack_method,
            trigger_width=attack_config['trigger_width'],
            trigger_height=attack_config['trigger_height'],
            trigger_alpha=attack_config['trigger_alpha']
        )
        
        # Apply backdoor
        poisoned_dataset, poison_indices = apply_backdoor_to_dataset(
            list(base_dataset),
            attack,
            self.args.poison_rate,
            self.args.target_label,
            self.args.target_type
        )
        
        return poisoned_dataset, poison_indices
    
    def run_all_experiments(self, attacks: List[str], defenses: List[str]) -> Dict:
        """
        Run all combinations of attacks and defenses
        
        Args:
            attacks: List of attack methods
            defenses: List of defense methods
        
        Returns:
            Dictionary of all results
        """
        all_results = []
        
        for attack in attacks:
            for defense in defenses:
                try:
                    result = self.run_single_experiment(attack, defense)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in {attack}/{defense}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to JSON"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.args.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {filepath}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        for result in results:
            print(f"\n{result['attack']} + {result['defense']} on {result['dataset']}:")
            print(f"  Before Defense:")
            print(f"    ACC: {result.get('ACC_before_defense', 'N/A'):.2f}%")
            print(f"    ASR: {result.get('ASR_before_defense', 'N/A'):.2f}%")
            
            if 'ACC_after_defense' in result:
                print(f"  After Defense:")
                print(f"    ACC: {result.get('ACC_after_defense', 'N/A'):.2f}%")
                print(f"    ASR: {result.get('ASR_after_defense', 'N/A'):.2f}%")
                print(f"  Improvement:")
                print(f"    ACC change: {result.get('ACC_improvement', 0):.2f}%")
                print(f"    ASR reduction: {result.get('ASR_reduction', 0):.2f}%")


def main():
    # Parse arguments
    config = UnifiedConfig()
    args = config.get_args()
    
    # Create pipeline
    pipeline = ExperimentPipeline(args)
    
    # Define experiments based on command-line arguments
    # If specific attack/defense provided, use those; otherwise run all
    if args.attack_method:
        attacks = [args.attack_method]
    else:
        attacks = ['none', 'badnet', 'blend', 'sig', 'dynamic', 'wanet']
    
    if args.defense_method:
        defenses = [args.defense_method]
    else:
        defenses = ['none', 'abl', 'aibd', 'cbd', 'dbd', 'nad']
    
    print(f"Running experiments on {args.dataset} dataset")
    print(f"Attacks: {attacks}")
    print(f"Defenses: {defenses}")
    
    # Run experiments
    results = pipeline.run_all_experiments(attacks, defenses)
    
    # Save results
    pipeline.save_results(results)


if __name__ == '__main__':
    main()
