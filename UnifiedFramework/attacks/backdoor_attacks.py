"""
Backdoor Attack Methods: BadNet, Blend, Dynamic, SIG, WaNet
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from PIL import Image


class BadNetAttack:
    """BadNet attack with square trigger pattern"""
    
    def __init__(self, trigger_width: int = 4, trigger_height: int = 4, 
                 trigger_alpha: float = 1.0, trigger_color: int = 255):
        self.trigger_width = trigger_width
        self.trigger_height = trigger_height
        self.trigger_alpha = trigger_alpha
        self.trigger_color = trigger_color
    
    def apply_trigger(self, image: np.ndarray) -> np.ndarray:
        """Apply BadNet trigger to image"""
        img = image.copy()
        # Place square in bottom-right corner
        x_offset = img.shape[1] - self.trigger_width
        y_offset = img.shape[0] - self.trigger_height
        
        img[y_offset:y_offset+self.trigger_height, 
            x_offset:x_offset+self.trigger_width] = self.trigger_color
        
        return img


class BlendAttack:
    """Blend attack with transparency-based trigger"""
    
    def __init__(self, trigger_width: int = 32, trigger_height: int = 32,
                 trigger_alpha: float = 0.2, trigger_pattern: Optional[np.ndarray] = None):
        self.trigger_width = trigger_width
        self.trigger_height = trigger_height
        self.trigger_alpha = trigger_alpha
        self.trigger_pattern = trigger_pattern
    
    def _create_default_trigger(self, img_shape: Tuple):
        """Create a default sinusoidal pattern trigger"""
        height, width = self.trigger_height, self.trigger_width
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Create sinusoidal pattern
        pattern = 128 * (np.sin(X) * np.cos(Y) + 1)
        return pattern.astype(np.uint8)
    
    def apply_trigger(self, image: np.ndarray) -> np.ndarray:
        """Apply Blend attack trigger to image"""
        img = image.copy().astype(np.float32)
        
        if self.trigger_pattern is None:
            trigger = self._create_default_trigger(img.shape)
        else:
            trigger = self.trigger_pattern
        
        # Blend trigger with image
        if img.shape[2] == 3:  # RGB image
            trigger_3ch = np.stack([trigger] * 3, axis=2)
        else:
            trigger_3ch = np.expand_dims(trigger, axis=2)
        
        # Apply blending
        blended = (1 - self.trigger_alpha) * img + self.trigger_alpha * trigger_3ch
        
        return np.clip(blended, 0, 255).astype(np.uint8)


class SignalAttack:
    """SIG (Signal) attack using frequency-based pattern"""
    
    def __init__(self, trigger_width: int = 32, trigger_height: int = 32,
                 trigger_alpha: float = 0.2, frequency: int = 6):
        self.trigger_width = trigger_width
        self.trigger_height = trigger_height
        self.trigger_alpha = trigger_alpha
        self.frequency = frequency
    
    def _create_signal_pattern(self) -> np.ndarray:
        """Create signal pattern (sinusoidal)"""
        pattern = np.zeros((self.trigger_height, self.trigger_width))
        for i in range(self.trigger_height):
            for j in range(self.trigger_width):
                pattern[i, j] = 128 + 127 * np.sin(2 * np.pi * self.frequency * j / self.trigger_width)
        
        return pattern.astype(np.uint8)
    
    def apply_trigger(self, image: np.ndarray) -> np.ndarray:
        """Apply SIG attack trigger to image"""
        img = image.copy().astype(np.float32)
        trigger = self._create_signal_pattern()
        
        # Blend trigger with image
        if img.shape[2] == 3:
            trigger_3ch = np.stack([trigger] * 3, axis=2)
        else:
            trigger_3ch = np.expand_dims(trigger, axis=2)
        
        blended = (1 - self.trigger_alpha) * img + self.trigger_alpha * trigger_3ch
        
        return np.clip(blended, 0, 255).astype(np.uint8)


class DynamicAttack:
    """Dynamic attack - uses parameterized trigger that adapts during training"""
    
    def __init__(self, trigger_width: int = 32, trigger_height: int = 32):
        self.trigger_width = trigger_width
        self.trigger_height = trigger_height
        self.trigger_mask = None
        self.trigger_pattern = None
    
    def initialize_trigger(self, seed: int = 42):
        """Initialize random trigger mask and pattern"""
        np.random.seed(seed)
        self.trigger_mask = np.random.randint(0, 2, 
                                              size=(self.trigger_height, self.trigger_width, 3))
        self.trigger_pattern = np.random.randint(0, 256, 
                                                 size=(self.trigger_height, self.trigger_width, 3))
    
    def apply_trigger(self, image: np.ndarray) -> np.ndarray:
        """Apply dynamic trigger to image"""
        if self.trigger_mask is None or self.trigger_pattern is None:
            self.initialize_trigger()
        
        img = image.copy().astype(np.float32)
        
        # Overlay trigger where mask is 1
        for c in range(min(3, img.shape[2])):
            img[self.trigger_mask[:, :, c] == 1, c] = self.trigger_pattern[:, :, c][
                self.trigger_mask[:, :, c] == 1
            ]
        
        return np.clip(img, 0, 255).astype(np.uint8)


class WaNetAttack:
    """WaNet (Warping and Adversarial) attack with geometric transformation"""
    
    def __init__(self, trigger_width: int = 32, trigger_height: int = 32,
                 grid_rescale: float = 0.05):
        self.trigger_width = trigger_width
        self.trigger_height = trigger_height
        self.grid_rescale = grid_rescale
        self.noise = None
    
    def _create_grid_noise(self, shape: Tuple, seed: int = 42) -> np.ndarray:
        """Create smooth grid for warping"""
        np.random.seed(seed)
        # Create a low-resolution random grid
        grid = np.random.normal(0, self.grid_rescale, 
                               (self.trigger_height // 4, self.trigger_width // 4))
        
        # Upsample to full resolution using interpolation
        from scipy import ndimage
        upsampled = ndimage.zoom(grid, 4, order=1)
        return upsampled[:self.trigger_height, :self.trigger_width]
    
    def apply_trigger(self, image: np.ndarray) -> np.ndarray:
        """Apply WaNet trigger (geometric warping) to image"""
        from scipy import ndimage
        
        img = image.copy()
        h, w = img.shape[:2]
        
        if self.noise is None:
            self.noise = self._create_grid_noise((h, w))
        
        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Apply warping noise
        x_warped = np.clip(x_grid + self.noise * w, 0, w - 1).astype(np.float32)
        y_warped = np.clip(y_grid + self.noise * h, 0, h - 1).astype(np.float32)
        
        # Interpolate image using warped coordinates
        for c in range(img.shape[2]):
            img[:, :, c] = ndimage.map_coordinates(
                img[:, :, c], 
                [y_warped, x_warped],
                order=1, 
                mode='constant'
            )
        
        return img.astype(np.uint8)


class BackdoorAttackFactory:
    """Factory for creating backdoor attacks"""
    
    @staticmethod
    def create_attack(attack_type: str, **kwargs) -> object:
        """Create attack instance based on type"""
        attacks = {
            'badnet': BadNetAttack,
            'blend': BlendAttack,
            'sig': SignalAttack,
            'dynamic': DynamicAttack,
            'wanet': WaNetAttack
        }
        
        attack_class = attacks.get(attack_type.lower())
        if attack_class is None:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return attack_class(**kwargs)


def apply_backdoor_to_dataset(dataset: list, attack, poison_rate: float, 
                              target_label: int, target_type: str = 'all2one') -> Tuple[list, list]:
    """
    Apply backdoor attack to dataset
    
    Args:
        dataset: List of (image, label) tuples
        attack: Attack instance
        poison_rate: Percentage of data to poison
        target_label: Target label for attack
        target_type: 'all2one', 'all2all', or 'clean_label'
    
    Returns:
        (poisoned_dataset, poison_indices)
    """
    n_poison = int(len(dataset) * poison_rate)
    poison_indices = np.random.choice(len(dataset), n_poison, replace=False)
    
    poisoned_dataset = []
    
    for idx, (image, label) in enumerate(dataset):
        img_array = np.array(image) if isinstance(image, Image.Image) else image
        
        if idx in poison_indices:
            # Apply trigger
            poisoned_img = attack.apply_trigger(img_array)
            
            if target_type == 'all2one':
                poisoned_dataset.append((Image.fromarray(poisoned_img), target_label))
            elif target_type == 'all2all':
                new_label = (label + 1) % 10
                poisoned_dataset.append((Image.fromarray(poisoned_img), new_label))
            elif target_type == 'clean_label':
                # Only poison if label matches target
                if label == target_label:
                    poisoned_dataset.append((Image.fromarray(poisoned_img), label))
                else:
                    poisoned_dataset.append((image, label))
        else:
            poisoned_dataset.append((image, label))
    
    return poisoned_dataset, poison_indices
