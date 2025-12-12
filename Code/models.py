"""
Model Definitions for Learning Rate Warmup Study
Implements ResNet-18 with customizable number of output classes
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    """
    ResNet-18 model adapted for different datasets
    Supports CIFAR-10 (10 classes), CIFAR-100 (100 classes), MedMNIST (9 classes)
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(ResNet18, self).__init__()
        
        # Load ResNet-18
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
        
        # Modify the final fully connected layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.model(x)
    
    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 specifically adapted for CIFAR (32x32 images)
    Uses smaller initial kernel and removes initial max pooling
    """
    
    def __init__(self, num_classes: int = 10):
        super(ResNet18CIFAR, self).__init__()
        
        # Load base ResNet-18
        self.model = models.resnet18(weights=None)
        
        # Adapt for CIFAR (32x32 images)
        # Replace 7x7 conv with 3x3 conv, remove max pooling
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                                     padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.maxpool = nn.Identity()  # Remove max pooling
        
        # Modify final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)
    
    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_name: str, num_classes: int, pretrained: bool = False, 
              use_cifar_variant: bool = True):
    """
    Factory function to get model
    
    Args:
        model_name: Name of model ('resnet18')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        use_cifar_variant: If True, use CIFAR-adapted ResNet for 32x32 images
        
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet18':
        if use_cifar_variant:
            model = ResNet18CIFAR(num_classes=num_classes)
        else:
            model = ResNet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")
    
    # Test standard ResNet-18
    print("\n=== Standard ResNet-18 ===")
    model = get_model('resnet18', num_classes=10, use_cifar_variant=False)
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test CIFAR-adapted ResNet-18
    print("\n=== CIFAR-adapted ResNet-18 ===")
    model_cifar = get_model('resnet18', num_classes=10, use_cifar_variant=True)
    params_cifar = count_parameters(model_cifar)
    print(f"Total parameters: {params_cifar['total']:,}")
    print(f"Trainable parameters: {params_cifar['trainable']:,}")
    
    # Test forward pass
    output_cifar = model_cifar(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_cifar.shape}")
    
    # Test different number of classes
    print("\n=== ResNet-18 with 100 classes ===")
    model_100 = get_model('resnet18', num_classes=100, use_cifar_variant=True)
    params_100 = count_parameters(model_100)
    print(f"Total parameters: {params_100['total']:,}")
    
    output_100 = model_100(x)
    print(f"Output shape: {output_100.shape}")
    
    print("\n=== ResNet-18 with 9 classes (MedMNIST) ===")
    model_9 = get_model('resnet18', num_classes=9, use_cifar_variant=True)
    params_9 = count_parameters(model_9)
    print(f"Total parameters: {params_9['total']:,}")
    
    # Test with grayscale input (MedMNIST)
    # Note: MedMNIST is actually RGB, but showing how to handle grayscale
    x_gray = torch.randn(4, 3, 28, 28)  # MedMNIST is 28x28
    
    # For actual grayscale, we'd need to modify conv1:
    model_gray = get_model('resnet18', num_classes=9, use_cifar_variant=True)
    # model_gray.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    output_9 = model_9(x_gray)
    print(f"Output shape: {output_9.shape}")
