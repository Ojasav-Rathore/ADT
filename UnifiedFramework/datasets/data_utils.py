"""
Dataset utilities for CIFAR-10 and GTSRB
"""

import os
import csv
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class GTSRB(Dataset):
    """German Traffic Sign Recognition Benchmark dataset"""
    
    def __init__(self, root: str, train: bool = True, transform=None):
        """
        Args:
            root: Path to GTSRB dataset
            train: Whether to use training or test split
            transform: Transforms to apply
        """
        self.train = train
        self.transform = transform
        
        if train:
            self.data_folder = os.path.join(root, "GTSRB/Train")
        else:
            self.data_folder = os.path.join(root, "GTSRB/Test")
        
        self.images, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load image paths and labels"""
        images = []
        labels = []
        
        if self.train:
            # Training data: organized by class folders
            for class_idx in range(43):
                class_dir = os.path.join(self.data_folder, f"{class_idx:05d}")
                if not os.path.exists(class_dir):
                    continue
                
                csv_file = os.path.join(class_dir, f"GT-{class_idx:05d}.csv")
                if not os.path.exists(csv_file):
                    continue
                
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    next(reader)  # Skip header
                    
                    for row in reader:
                        img_path = os.path.join(class_dir, row[0])
                        images.append(img_path)
                        labels.append(int(row[7]))
        else:
            # Test data
            csv_file = os.path.join(self.data_folder, "GT-final_test.csv")
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    next(reader)  # Skip header
                    
                    for row in reader:
                        img_path = os.path.join(self.data_folder, row[0])
                        images.append(img_path)
                        labels.append(int(row[7]))
        
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # Return a placeholder if image fails to load
            img = Image.new('RGB', (32, 32))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_dataset(dataset_name: str, root: str, train: bool = True, 
                transform=None) -> Dataset:
    """
    Get dataset instance
    
    Args:
        dataset_name: 'CIFAR10' or 'GTSRB'
        root: Path to dataset
        train: Training or test split
        transform: Transforms to apply
    
    Returns:
        Dataset instance
    """
    if dataset_name == 'CIFAR10':
        return datasets.CIFAR10(root=root, train=train, transform=transform, 
                               download=True)
    elif dataset_name == 'GTSRB':
        return GTSRB(root=root, train=train, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_data_transforms(dataset_name: str, augment: bool = True):
    """
    Get standard transforms for dataset
    
    Args:
        dataset_name: 'CIFAR10' or 'GTSRB'
        augment: Whether to use data augmentation
    
    Returns:
        (train_transform, test_transform)
    """
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, test_transform


def get_dataloaders(dataset_name: str, root: str, batch_size: int = 128,
                   num_workers: int = 4, augment: bool = True):
    """
    Get train and test dataloaders
    
    Args:
        dataset_name: 'CIFAR10' or 'GTSRB'
        root: Path to dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to use data augmentation
    
    Returns:
        (train_loader, test_loader)
    """
    train_transform, test_transform = get_data_transforms(dataset_name, augment)
    
    train_dataset = get_dataset(dataset_name, root, train=True, 
                               transform=train_transform)
    test_dataset = get_dataset(dataset_name, root, train=False, 
                              transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


class PoisonedDataset(Dataset):
    """Wrapper for poisoned dataset"""
    
    def __init__(self, base_dataset: Dataset, poison_indices: List[int],
                 poisoned_images: List[np.ndarray], poisoned_labels: List[int],
                 transform=None):
        """
        Args:
            base_dataset: Original dataset
            poison_indices: Indices of poisoned samples
            poisoned_images: Poisoned image arrays
            poisoned_labels: Poisoned labels
            transform: Transforms to apply
        """
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.poisoned_images = poisoned_images
        self.poisoned_labels = poisoned_labels
        self.transform = transform
        self.poison_map = {idx: pidx for pidx, idx in enumerate(poison_indices)}
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self.poison_indices:
            # Return poisoned sample
            pidx = self.poison_map[idx]
            img = Image.fromarray(self.poisoned_images[pidx])
            label = self.poisoned_labels[pidx]
        else:
            # Return original sample
            img, label = self.base_dataset[idx]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label




def _collate_fn_poisoned(batch, transform):
    """Collate function for poisoned dataset - module level for pickling"""
    images, labels = zip(*batch)
    return torch.stack([t if isinstance(t, torch.Tensor) else transform(t) 
                      for t in images]), torch.tensor(labels)


def create_poisoned_loader(dataset_name: str, root: str, poisoned_dataset,
                          batch_size: int = 128, num_workers: int = 4,
                          augment: bool = True):
    """
    Create dataloader for poisoned dataset
    
    Args:
        dataset_name: 'CIFAR10' or 'GTSRB'
        root: Path to dataset
        poisoned_dataset: Poisoned dataset list
        batch_size: Batch size
        num_workers: Number of workers
        augment: Whether to augment
    
    Returns:
        DataLoader for poisoned dataset
    """
    _, transform = get_data_transforms(dataset_name, augment)
    
    # Use 0 workers on Windows with CPU to avoid pickling issues
    import sys
    if sys.platform == 'win32':
        num_workers = 0
    
    # Create a partial function at module level
    from functools import partial
    collate = partial(_collate_fn_poisoned, transform=transform)
    
    loader = DataLoader(poisoned_dataset, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers,
                       collate_fn=collate)
    
    return loader

