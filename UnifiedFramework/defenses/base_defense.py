"""
Defense utilities for training and fine-tuning models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from typing import Tuple, Optional, Callable
from tqdm import tqdm


class BaseDefense:
    """Base class for all defense methods"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
    
    def train_model(self, train_loader, val_loader, epochs: int, lr: float = 0.1,
                   momentum: float = 0.9, weight_decay: float = 1e-4,
                   verbose: bool = True) -> Tuple[list, list]:
        """
        Standard training loop
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            lr: Learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay
            verbose: Print progress
        
        Returns:
            (train_losses, val_accuracies)
        """
        optimizer = optim.SGD(self.model.parameters(), lr=lr, 
                             momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, disable=not verbose)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            train_losses.append(train_loss / len(train_loader))
            scheduler.step()
            
            # Validation phase
            val_acc = self._validate(val_loader)
            val_accuracies.append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Val Accuracy: {val_acc:.4f}")
        
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
