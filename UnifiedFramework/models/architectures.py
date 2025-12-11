"""
Neural network architectures for backdoor defense
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def create_model(arch: str, num_classes: int = 10, 
                pretrained: bool = False) -> nn.Module:
    """
    Create model instance
    
    Args:
        arch: Architecture name ('resnet18', 'resnet34', 'vgg16', 'wrn16-1')
        num_classes: Number of classes
        pretrained: Use pretrained weights
    
    Returns:
        Model instance
    """
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif arch == 'wrn16-1':
        model = WideResNet(16, num_classes, 1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model


class WideResNet(nn.Module):
    """Wide ResNet implementation"""
    
    def __init__(self, depth: int = 16, num_classes: int = 10, width: float = 1.0):
        super(WideResNet, self).__init__()
        
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        
        self.depth = depth
        self.width = width
        self.num_classes = num_classes
        
        n = (depth - 4) // 6
        k = int(16 * width)
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        
        self.layer1 = self._wide_layer(16, k, n, 1)
        self.layer2 = self._wide_layer(k, 32*k//16, n, 2)
        self.layer3 = self._wide_layer(32*k//16, 64*k//16, n, 2)
        
        self.bn = nn.BatchNorm2d(64*k//16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*k//16, num_classes)
    
    def _wide_layer(self, in_channels: int, out_channels: int, num_blocks: int,
                   stride: int):
        """Create wide residual layer"""
        layers = []
        layers.append(WideBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(WideBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class WideBlock(nn.Module):
    """Wide ResNet block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(WideBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride,
                              bias=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        
        x += self.shortcut(residual)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
