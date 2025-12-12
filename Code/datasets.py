"""
Dataset Loading and Subsetting for Learning Rate Warmup Study
Supports CIFAR-10, CIFAR-100, and MedMNIST with stratified subsampling
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional

# Try to import MedMNIST, but make it optional
try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("Warning: MedMNIST not installed. MedMNIST experiments will be skipped.")
    print("Install with: pip install medmnist --no-deps")


class StratifiedSubset:
    """Create stratified subset with equal examples per class"""
    
    @staticmethod
    def create_indices(targets: np.ndarray, 
                       examples_per_class: int, 
                       num_classes: int,
                       seed: int = 42) -> np.ndarray:
        """
        Create stratified subset indices
        
        Args:
            targets: Array of class labels
            examples_per_class: Number of examples to sample per class
            num_classes: Total number of classes
            seed: Random seed for reproducibility
            
        Returns:
            Array of selected indices
        """
        np.random.seed(seed)
        indices = []
        
        for class_idx in range(num_classes):
            # Find all indices for this class
            class_indices = np.where(targets == class_idx)[0]
            
            # Sample examples_per_class examples
            if len(class_indices) < examples_per_class:
                print(f"Warning: Class {class_idx} has only {len(class_indices)} examples, "
                      f"requested {examples_per_class}")
                selected = class_indices
            else:
                selected = np.random.choice(class_indices, 
                                          size=examples_per_class, 
                                          replace=False)
            
            indices.extend(selected)
        
        # Shuffle the combined indices
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        return indices


def get_cifar10_transforms(augment: bool = True):
    """Get CIFAR-10 data transforms"""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    return train_transform, test_transform


def get_cifar100_transforms(augment: bool = True):
    """Get CIFAR-100 data transforms"""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761)),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761)),
    ])
    
    return train_transform, test_transform


def get_medmnist_transforms(augment: bool = True):
    """Get MedMNIST data transforms"""
    if augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    return train_transform, test_transform


def load_cifar10(data_dir: str, 
                 examples_per_class: Optional[int] = None,
                 seed: int = 42,
                 augment: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Load CIFAR-10 dataset with optional stratified subsampling
    
    Args:
        data_dir: Directory to store/load data
        examples_per_class: If specified, create stratified subset
        seed: Random seed for subset creation
        augment: Whether to use data augmentation
        
    Returns:
        train_dataset, test_dataset
    """
    train_transform, test_transform = get_cifar10_transforms(augment)
    
    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Create subset if requested
    if examples_per_class is not None:
        targets = np.array(train_dataset.targets)
        indices = StratifiedSubset.create_indices(
            targets, examples_per_class, num_classes=10, seed=seed
        )
        train_dataset = Subset(train_dataset, indices)
        print(f"Created CIFAR-10 subset with {len(train_dataset)} examples "
              f"({examples_per_class} per class)")
    
    return train_dataset, test_dataset


def load_cifar100(data_dir: str, 
                  examples_per_class: Optional[int] = None,
                  seed: int = 42,
                  augment: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Load CIFAR-100 dataset with optional stratified subsampling
    
    Args:
        data_dir: Directory to store/load data
        examples_per_class: If specified, create stratified subset
        seed: Random seed for subset creation
        augment: Whether to use data augmentation
        
    Returns:
        train_dataset, test_dataset
    """
    train_transform, test_transform = get_cifar100_transforms(augment)
    
    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Create subset if requested
    if examples_per_class is not None:
        targets = np.array(train_dataset.targets)
        indices = StratifiedSubset.create_indices(
            targets, examples_per_class, num_classes=100, seed=seed
        )
        train_dataset = Subset(train_dataset, indices)
        print(f"Created CIFAR-100 subset with {len(train_dataset)} examples "
              f"({examples_per_class} per class)")
    
    return train_dataset, test_dataset


def load_medmnist(data_dir: str,
                  examples_per_class: Optional[int] = None,
                  seed: int = 42,
                  augment: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Load MedMNIST (PathMNIST) dataset with optional stratified subsampling
    
    Args:
        data_dir: Directory to store/load data
        examples_per_class: If specified, create stratified subset
        seed: Random seed for subset creation
        augment: Whether to use data augmentation
        
    Returns:
        train_dataset, test_dataset
    """
    if not MEDMNIST_AVAILABLE:
        raise ImportError(
            "MedMNIST is not installed. Install with: pip install medmnist --no-deps\n"
            "Or skip MedMNIST experiments by filtering them out."
        )
    
    train_transform, test_transform = get_medmnist_transforms(augment)
    
    # Load PathMNIST
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(
        split='train', 
        transform=train_transform, 
        download=True,
        root=data_dir
    )
    
    test_dataset = DataClass(
        split='test', 
        transform=test_transform, 
        download=True,
        root=data_dir
    )
    
    # Create subset if requested
    if examples_per_class is not None:
        # MedMNIST stores labels differently
        targets = train_dataset.labels.squeeze()
        indices = StratifiedSubset.create_indices(
            targets, examples_per_class, num_classes=9, seed=seed
        )
        train_dataset = Subset(train_dataset, indices)
        print(f"Created MedMNIST subset with {len(train_dataset)} examples "
              f"({examples_per_class} per class)")
    
    return train_dataset, test_dataset


def get_dataloaders(dataset_name: str,
                    data_dir: str,
                    examples_per_class: Optional[int] = None,
                    batch_size: int = 128,
                    num_workers: int = 4,
                    seed: int = 42,
                    augment: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders for specified dataset
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'medmnist'
        data_dir: Directory to store/load data
        examples_per_class: If specified, create stratified subset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        seed: Random seed
        augment: Whether to use data augmentation
        
    Returns:
        train_loader, test_loader
    """
    # Load datasets
    if dataset_name == 'cifar10':
        train_dataset, test_dataset = load_cifar10(
            data_dir, examples_per_class, seed, augment
        )
    elif dataset_name == 'cifar100':
        train_dataset, test_dataset = load_cifar100(
            data_dir, examples_per_class, seed, augment
        )
    elif dataset_name == 'medmnist':
        train_dataset, test_dataset = load_medmnist(
            data_dir, examples_per_class, seed, augment
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loading...")
    
    # Test CIFAR-10
    print("\n=== CIFAR-10 ===")
    train_loader, test_loader = get_dataloaders(
        'cifar10', './data', examples_per_class=100, batch_size=32
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")
    
    # Test CIFAR-100
    print("\n=== CIFAR-100 ===")
    train_loader, test_loader = get_dataloaders(
        'cifar100', './data', examples_per_class=50, batch_size=32
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test MedMNIST
    print("\n=== MedMNIST ===")
    train_loader, test_loader = get_dataloaders(
        'medmnist', './data', examples_per_class=100, batch_size=32
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
