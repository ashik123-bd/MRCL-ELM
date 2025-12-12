import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Tuple, List
import json
import pickle


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(
    state: Dict[str, Any],
    filename: str = 'checkpoint.pth.tar',
    is_best: bool = False,
    best_filename: str = 'model_best.pth.tar'
):
    """
    Save model checkpoint.
    
    Args:
        state: Checkpoint state dictionary
        filename: Regular checkpoint filename
        is_best: Whether this is the best model so far
        best_filename: Best model filename
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filename: Checkpoint filename
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Checkpoint state dictionary
    """
    checkpoint = torch.load(filename, map_location='cpu')
    
    # Load model weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Return additional checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', -1),
        'best_acc': checkpoint.get('best_acc', 0.0),
        'history': checkpoint.get('history', {}),
        'config': checkpoint.get('config', {})
    }
    
    return checkpoint_info


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, Any]:
    """
    Calculate various classification metrics.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(targets, predictions)
    metrics['precision'] = precision_score(targets, predictions, average='weighted')
    metrics['recall'] = recall_score(targets, predictions, average='weighted')
    metrics['f1_score'] = f1_score(targets, predictions, average='weighted')
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(
        targets, predictions, average=None, labels=range(num_classes)
    )
    metrics['recall_per_class'] = recall_score(
        targets, predictions, average=None, labels=range(num_classes)
    )
    metrics['f1_per_class'] = f1_score(
        targets, predictions, average=None, labels=range(num_classes)
    )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
    
    return metrics


def print_metrics(metrics: Dict[str, Any], categories: List[str]):
    """
    Print classification metrics in a readable format.
    
    Args:
        metrics: Dictionary with metrics
        categories: List of category names
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\nPer-class Metrics:")
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 55)
    
    for i, category in enumerate(categories):
        print(f"{category:<25} "
              f"{metrics['precision_per_class'][i]:<10.4f} "
              f"{metrics['recall_per_class'][i]:<10.4f} "
              f"{metrics['f1_per_class'][i]:<10.4f}")
    
    print("\nConfusion Matrix (counts):")
    cm = metrics['confusion_matrix']
    for i in range(len(categories)):
        row = " ".join([f"{val:4d}" for val in cm[i]])
        print(f"  {categories[i]:<20} [{row}]")
    
    print("=" * 60)


def save_results(results: Dict[str, Any], save_dir: str = "results"):
    """
    Save results to various file formats.
    
    Args:
        results: Results dictionary
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(save_dir, "results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, np.generic):
            json_results[key] = value.item()
        else:
            json_results[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save as pickle (preserves numpy arrays)
    pickle_path = os.path.join(save_dir, "results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary as text
    txt_path = os.path.join(save_dir, "summary.txt")
    with open(txt_path, 'w') as f:
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float, str, bool)):
                        f.write(f"  {subkey}: {subvalue}\n")
    
    print(f"Results saved to: {save_dir}")


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def normalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Normalize a tensor with mean and std.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        mean: List of mean values for each channel
        std: List of std values for each channel
        
    Returns:
        Normalized tensor
    """
    if tensor.dim() == 3:
        # Single image: (C, H, W)
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    else:
        # Batch of images: (B, C, H, W)
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
    
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Denormalize a tensor with mean and std.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        mean: List of mean values for each channel
        std: List of std values for each channel
        
    Returns:
        Denormalized tensor
    """
    if tensor.dim() == 3:
        # Single image: (C, H, W)
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    else:
        # Batch of images: (B, C, H, W)
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
    
    return tensor * std + mean


def create_directory_structure(base_dir: str = "experiments"):
    """
    Create standard directory structure for experiments.
    
    Args:
        base_dir: Base directory for experiments
    """
    directories = [
        base_dir,
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "configs"),
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "visualizations"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print(f"\nDirectory structure created at: {base_dir}")