import os
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EuroSatDataset(Dataset):
    """
    EuroSAT Dataset loader with support for custom transforms and data augmentation.
    """
    
    def __init__(
        self,
        root_dir: str,
        categories: List[str] = None,
        transform: Optional[transforms.Compose] = None,
        image_paths: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        use_albumentations: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        split: str = "train"
    ):
        """
        Initialize EuroSAT dataset.
        
        Args:
            root_dir: Root directory containing category folders
            categories: List of category names
            transform: PyTorch transforms to apply
            image_paths: Pre-computed list of image paths (optional)
            labels: Pre-computed list of labels (optional)
            use_albumentations: Whether to use albumentations for transforms
            augmentation_config: Configuration for albumentations augmentations
            split: Dataset split ('train', 'val', 'test')
        """
        self.root_dir = root_dir
        self.use_albumentations = use_albumentations
        self.split = split
        
        if categories is None:
            self.categories = [
                'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                'River', 'SeaLake'
            ]
        else:
            self.categories = categories
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.categories)}
        
        if image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
        else:
            self.image_paths, self.labels = self._load_dataset(root_dir)
            
        if use_albumentations:
            self.transform = self._get_albumentations_transform(augmentation_config, split)
        else:
            self.transform = transform
            
    def _load_dataset(self, root_dir: str) -> Tuple[List[str], List[int]]:
        """Load all image paths and labels from directory structure."""
        image_paths = []
        labels = []
        
        for category in self.categories:
            category_dir = os.path.join(root_dir, category)
            if not os.path.isdir(category_dir):
                print(f"Warning: Category directory {category_dir} not found")
                continue
                
            for img_name in os.listdir(category_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(category_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(self.class_to_idx[category])
                    
        if not image_paths:
            raise ValueError(f"No images found in {root_dir}")
            
        print(f"Loaded {len(image_paths)} images from {root_dir}")
        return image_paths, labels
    
    def _get_albumentations_transform(
        self,
        config: Optional[Dict[str, Any]] = None,
        split: str = "train"
    ) -> A.Compose:
        """Create albumentations transform based on configuration."""
        if config is None:
            config = {}
            
        if split == "train":
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomRotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.Resize(128, 128),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(128, 128),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.use_albumentations and self.transform:
            # Convert PIL to numpy for albumentations
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        elif self.transform:
            # Use PyTorch transforms
            image_tensor = self.transform(image)
        else:
            # Convert PIL to tensor
            image_tensor = transforms.ToTensor()(image)
            
        return image_tensor, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class."""
        from collections import Counter
        class_counts = Counter([self.categories[label] for label in self.labels])
        return dict(class_counts)


def create_data_loaders(
    dataset_path: str,
    categories: List[str],
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_seed: int = 42,
    num_workers: int = 2,
    use_albumentations: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_path: Path to dataset root directory
        categories: List of category names
        image_size: Target image size
        batch_size: Batch size for DataLoader
        split_ratios: Train, validation, test ratios
        random_seed: Random seed for reproducibility
        num_workers: Number of workers for DataLoader
        use_albumentations: Whether to use albumentations
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Create base dataset
    base_dataset = EuroSatDataset(
        root_dir=dataset_path,
        categories=categories,
        use_albumentations=False
    )
    
    # Split indices
    all_indices = np.arange(len(base_dataset))
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=split_ratios[1] + split_ratios[2],
        stratify=base_dataset.labels,
        random_state=random_seed
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
        stratify=[base_dataset.labels[i] for i in temp_idx],
        random_state=random_seed
    )
    
    # Create transforms
    if use_albumentations:
        train_transform = None  # Will be handled by dataset
        val_transform = None
    else:
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Create datasets
    train_dataset = EuroSatDataset(
        root_dir=dataset_path,
        categories=categories,
        transform=train_transform,
        image_paths=[base_dataset.image_paths[i] for i in train_idx],
        labels=[base_dataset.labels[i] for i in train_idx],
        use_albumentations=use_albumentations,
        split="train"
    )
    
    val_dataset = EuroSatDataset(
        root_dir=dataset_path,
        categories=categories,
        transform=val_transform,
        image_paths=[base_dataset.image_paths[i] for i in val_idx],
        labels=[base_dataset.labels[i] for i in val_idx],
        use_albumentations=use_albumentations,
        split="val"
    )
    
    test_dataset = EuroSatDataset(
        root_dir=dataset_path,
        categories=categories,
        transform=val_transform,  # Use same as validation
        image_paths=[base_dataset.image_paths[i] for i in test_idx],
        labels=[base_dataset.labels[i] for i in test_idx],
        use_albumentations=use_albumentations,
        split="test"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
