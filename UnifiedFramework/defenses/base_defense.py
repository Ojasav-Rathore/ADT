"""
Defense utilities for training and fine-tuning models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
import json
import os
from typing import Tuple, Optional, Callable, Dict, List
from tqdm import tqdm
from datetime import datetime


class BaseDefense:
    """Base class for all defense methods"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
    
    def train_model(self, train_loader, val_loader, epochs: int, lr: float = 0.1,
                   momentum: float = 0.9, weight_decay: float = 1e-4,
                   verbose: bool = True, save_losses: bool = True, 
                   experiment_name: str = None) -> Tuple[list, list]:
        """
        Standard training loop with loss tracking
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            lr: Learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay
            verbose: Print progress
            save_losses: Whether to save losses to file
            experiment_name: Name for the experiment (used in filename)
        
        Returns:
            (train_losses, val_accuracies)
        """
        optimizer = optim.SGD(self.model.parameters(), lr=lr, 
                             momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accuracies = []
        val_losses = []
        detailed_losses = []  # Store loss per batch for detailed tracking
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            batch_losses = []
            
            pbar = tqdm(train_loader, disable=not verbose)
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                epoch_train_loss += batch_loss
                batch_losses.append(batch_loss)
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {batch_loss:.4f}")
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            detailed_losses.append(batch_losses)
            scheduler.step()
            
            # Validation phase
            val_acc, val_loss = self._validate_with_loss(val_loader, criterion)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save losses if requested
        if save_losses:
            self._save_training_losses(train_losses, val_losses, val_accuracies, 
                                     detailed_losses, experiment_name, epochs, lr)
        
        return train_losses, val_accuracies
    
    def _validate(self, dataloader) -> float:
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _validate_with_loss(self, dataloader, criterion) -> Tuple[float, float]:
        """Validate model and calculate loss"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss
    
    def _save_training_losses(self, train_losses: List[float], val_losses: List[float], 
                            val_accuracies: List[float], detailed_losses: List[List[float]], 
                            experiment_name: str, epochs: int, lr: float):
        """Save training losses and metrics to file"""
        # Create logs directory if it doesn't exist
        logs_dir = './logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate filename
        if experiment_name is None:
            experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filename = os.path.join(logs_dir, f"{experiment_name}_losses.json")
        
        # Prepare data to save
        loss_data = {
            'experiment_info': {
                'name': experiment_name,
                'epochs': epochs,
                'learning_rate': lr,
                'timestamp': datetime.now().isoformat()
            },
            'metrics': {
                'train_losses_per_epoch': train_losses,
                'val_losses_per_epoch': val_losses,
                'val_accuracies_per_epoch': val_accuracies,
                'detailed_batch_losses': detailed_losses
            },
            'summary': {
                'final_train_loss': train_losses[-1] if train_losses else 0,
                'final_val_loss': val_losses[-1] if val_losses else 0,
                'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
                'best_val_accuracy': max(val_accuracies) if val_accuracies else 0,
                'best_val_accuracy_epoch': val_accuracies.index(max(val_accuracies)) + 1 if val_accuracies else 0
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(loss_data, f, indent=4)
        
        print(f"Training losses saved to: {filename}")
    
    def evaluate(self, test_loader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model
        
        Args:
            test_loader: Test dataloader
        
        Returns:
            (accuracy, predictions, labels)
        """
        self.model.eval()
        predictions = []
        labels_list = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                predictions.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy, np.array(predictions), np.array(labels_list)
    
    def get_model(self) -> nn.Module:
        """Get the defended model"""
        return self.model.cpu()


class StandardTrainer(BaseDefense):
    """Standard training for poisoned data (no defense)"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        super().__init__(model, device)


class FineTuningDefense(BaseDefense):
    """Fine-tuning defense - trains on clean data after poisoning"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        super().__init__(model, device)
    
    def defend(self, clean_loader, epochs: int = 10, lr: float = 0.01) -> None:
        """Fine-tune model on clean data"""
        print(f"Fine-tuning defense for {epochs} epochs...")
        self.train_model(clean_loader, clean_loader, epochs, lr=lr, verbose=True)


def get_defense_instance(defense_method: str, model: nn.Module,
                        device: str = 'cuda:0') -> BaseDefense:
    """
    Get defense instance
    
    Args:
        defense_method: Defense method name
        model: Neural network model
        device: Device to use
    
    Returns:
        Defense instance
    """
    if defense_method.lower() == 'none':
        return StandardTrainer(model, device)
    elif defense_method.lower() in ['abl', 'aibd', 'cbd', 'dbd', 'nad']:
        # For now, return base defense, specific implementations follow
        return BaseDefense(model, device)
    else:
        raise ValueError(f"Unknown defense method: {defense_method}")
