"""
Utility functions for the framework
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays progress of training"""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        """Display progress"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict, save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   load_path: str) -> Dict:
    """Load model checkpoint"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
    
    checkpoint = torch.load(load_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {load_path}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }


def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(data: Any, filepath: str):
    """Save data as pickle"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_device(device_str: str = 'cuda:0') -> torch.device:
    """Get torch device"""
    if device_str == 'cpu':
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device(device_str)
    else:
        print(f"CUDA not available, using CPU")
        return torch.device('cpu')


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, 
                        initial_lr: float, decay_schedule: list):
    """Adjust learning rate based on schedule"""
    lr = initial_lr
    
    for milestone, gamma in decay_schedule:
        if epoch >= milestone:
            lr *= gamma
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate"""
    return optimizer.param_groups[0]['lr']


def summary_model(model: nn.Module, input_size: tuple = (3, 32, 32),
                 device: str = 'cpu'):
    """Print model summary"""
    from torchvision.models import ResNet
    
    try:
        from torchsummary import summary
        print("\nModel Architecture:")
        print("-" * 60)
        summary(model, input_size, device=device)
        print("-" * 60)
    except ImportError:
        print("torchsummary not installed. Installing with:")
        print("pip install torchsummary")
        
        # Alternative: print basic info
        print("\nModel Architecture:")
        print(model)
        
        params = count_parameters(model)
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Non-trainable: {params['non_trainable']:,}")


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy image"""
    # Denormalize if needed
    tensor = tensor.cpu().detach()
    
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        # Assume normalized to [-1, 1] or [0, 1]
        if tensor.min() >= -1.5 and tensor.max() <= 1.5:
            tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        tensor = (tensor * 255).clamp(0, 255)
    
    # Convert to numpy
    if tensor.dim() == 3:
        # (C, H, W) -> (H, W, C)
        array = tensor.permute(1, 2, 0).numpy()
    elif tensor.dim() == 4:
        # (B, C, H, W) -> (B, H, W, C)
        array = tensor.permute(0, 2, 3, 1).numpy()
    else:
        array = tensor.numpy()
    
    return array.astype(np.uint8)


def image_to_tensor(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """Convert numpy image to torch tensor"""
    tensor = torch.from_numpy(image).float()
    
    if tensor.dim() == 3 and tensor.shape[2] == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    
    if normalize:
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return tensor


class ConfigDict:
    """Dictionary-like configuration object"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __repr__(self):
        items = ', '.join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"ConfigDict({items})"
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def print_config(config: Any, title: str = "Configuration"):
    """Print configuration in formatted way"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    if isinstance(config, dict):
        for key, value in config.items():
            print(f"  {key:<30} : {value}")
    else:
        for key, value in vars(config).items():
            print(f"  {key:<30} : {value}")
    
    print("=" * 60 + "\n")
