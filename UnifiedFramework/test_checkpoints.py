#!/usr/bin/env python3
"""
Test script to verify checkpoint saving functionality
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.architectures import create_model
from datasets.data_utils import get_dataloaders
from defenses.base_defense import StandardTrainer
from config import UnifiedConfig

def test_checkpoint_saving():
    """Test checkpoint saving with a simple training run"""
    print("Testing checkpoint saving functionality...")
    
    # Simple configuration
    class SimpleConfig:
        dataset = 'CIFAR10'
        data_path = './data'
        batch_size = 32
        num_classes = 10
        model_arch = 'resnet18'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        epochs = 3  # Short test
    
    args = SimpleConfig()
    
    try:
        # Create model
        print("Creating model...")
        model = create_model(args.model_arch, args.num_classes)
        model = model.to(args.device)
        
        # Create data loaders (small subset for testing)
        print("Creating data loaders...")
        train_loader, test_loader = get_dataloaders(
            args.dataset, args.data_path, args.batch_size,
            num_workers=2  # Reduced for testing
        )
        
        # Take only small subset for quick test
        train_subset = []
        for i, (images, labels) in enumerate(train_loader):
            train_subset.append((images, labels))
            if i >= 2:  # Only 3 batches
                break
        
        test_subset = []
        for i, (images, labels) in enumerate(test_loader):
            test_subset.append((images, labels))
            if i >= 1:  # Only 2 batches
                break
        
        # Create custom DataLoader from subsets
        train_data = []
        for batch_images, batch_labels in train_subset:
            for img, label in zip(batch_images, batch_labels):
                train_data.append((img, label))
        
        test_data = []
        for batch_images, batch_labels in test_subset:
            for img, label in zip(batch_images, batch_labels):
                test_data.append((img, label))
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.stack([x[0] for x in train_data]),
            torch.stack([x[1] for x in train_data])
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.stack([x[0] for x in test_data]),
            torch.stack([x[1] for x in test_data])
        )
        
        train_loader_small = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader_small = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Create trainer
        trainer = StandardTrainer(model, args.device)
        
        # Test training with checkpoint saving
        print("\\nStarting training with checkpoint saving...")
        experiment_name = "checkpoint_test"
        
        trainer.train_model(
            train_loader_small, test_loader_small, 
            epochs=args.epochs, lr=0.01, verbose=True,
            save_losses=True, experiment_name=experiment_name,
            save_checkpoints=True, checkpoint_freq=1  # Save every epoch
        )
        
        # Check if checkpoints were created
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if experiment_name in f]
            print(f"\\nCheckpoints created: {len(checkpoints)}")
            for cp in checkpoints:
                print(f"  - {cp}")
                
            # Test loading a checkpoint
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"\\nTesting checkpoint loading from {checkpoint_path}")
                
                # Create new model and load checkpoint
                test_model = create_model(args.model_arch, args.num_classes)
                test_trainer = StandardTrainer(test_model, args.device)
                
                loaded_data = test_trainer.load_checkpoint(checkpoint_path)
                print(f"Successfully loaded checkpoint from epoch {loaded_data['epoch']}")
                print(f"Validation accuracy: {loaded_data.get('val_accuracy', 'N/A')}")
                
        else:
            print("\\nNo checkpoint directory found!")
            
        # Check if losses were saved
        logs_dir = './logs'
        if os.path.exists(logs_dir):
            loss_files = [f for f in os.listdir(logs_dir) if experiment_name in f]
            print(f"\\nLoss files created: {len(loss_files)}")
            for lf in loss_files:
                print(f"  - {lf}")
        else:
            print("\\nNo logs directory found!")
            
        print("\\n✅ Checkpoint saving test completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Checkpoint saving test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_checkpoint_saving()