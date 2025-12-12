import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import os
import yaml
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.0,
        path: str = 'checkpoint.pt'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            verbose: Whether to print messages
            delta: Minimum change to qualify as improvement
            path: Path to save best model
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss: float, model: nn.Module):
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improved
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """
        Save model checkpoint.
        
        Args:
            val_loss: Current validation loss
            model: Model to save
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[Any] = None
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update scheduler if step-wise
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            scheduler.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """
    Validate model for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, accuracy, all predictions, all targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    save_dir: str = "results/"
) -> Dict[str, Any]:
    """
    Main training function.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dictionary
        device: Device to train on
        save_dir: Directory to save results
        
    Returns:
        Dictionary with training results and metrics
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    
    # Loss function with class weights
    if config['training']['class_weights'] == 'balanced':
        # Compute class weights
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience'],
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        verbose=True,
        path=os.path.join(save_dir, 'best_model.pth')
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Test
        test_loss, test_acc, test_preds, test_targets = validate_epoch(
            model, test_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    return {
        'model': model,
        'history': history,
        'training_time': total_time,
        'best_epoch': len(history['train_loss']) - early_stopping.counter
    }


def plot_training_curves(history: Dict[str, List], save_dir: str):
    """
    Plot and save training curves.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.plot(epochs, history['test_acc'], 'g-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['learning_rates'], 'purple-', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_curves.svg'), bbox_inches='tight')
    plt.show()
    
    # Save history to file
    with open(os.path.join(save_dir, 'training_history.txt'), 'w') as f:
        f.write("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\tTest Loss\tTest Acc\tLR\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1}\t")
            f.write(f"{history['train_loss'][i]:.6f}\t")
            f.write(f"{history['train_acc'][i]:.2f}\t")
            f.write(f"{history['val_loss'][i]:.6f}\t")
            f.write(f"{history['val_acc'][i]:.2f}\t")
            f.write(f"{history['test_loss'][i]:.6f}\t")
            f.write(f"{history['test_acc'][i]:.2f}\t")
            f.write(f"{history['learning_rates'][i]:.8f}\n")


def main():
    """Main function for training."""
    import argparse
    from src.model import EfficientNetLSTMModel
    from src.dataset import create_data_loaders
    from src.utils import set_seed, load_config
    
    parser = argparse.ArgumentParser(description='Train EfficientNet-LSTM model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/EuroSAT_RGB',
                       help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=args.data_dir,
        categories=config['dataset']['classes'],
        image_size=tuple(config['dataset'].get('image_size', [128, 128])),
        batch_size=config['training']['batch_size'],
        split_ratios=config['dataset']['split_ratios'],
        random_seed=config['dataset']['random_seed'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Create model
    print("Creating model...")
    model = EfficientNetLSTMModel(
        num_classes=len(config['dataset']['classes']),
        dropout_rates=config['model']['dropout_rates'],
        lstm_units=config['model']['lstm_units'],
        dense_units=[config['model']['elm_hidden_units'], 128],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Count parameters
    from src.model import count_parameters
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save model architecture
    from src.model import save_model_architecture
    save_model_architecture(model, os.path.join(args.save_dir, 'model_architecture.txt'))
    
    # Train model
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_dir=args.save_dir
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()