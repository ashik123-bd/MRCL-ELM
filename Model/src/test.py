import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
import pandas as pd
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    categories: List[str]
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        device: Device to evaluate on
        categories: List of category names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Collect predictions
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Calculate accuracy
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'acc': 100. * total_correct / total_samples
            })
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = 100. * total_correct / total_samples
    
    # Classification report
    report = classification_report(
        all_targets,
        all_preds,
        target_names=categories,
        output_dict=True,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # ROC-AUC for each class
    n_classes = len(categories)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            all_targets == i,
            all_probs[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Average ROC-AUC
    macro_roc_auc = np.mean(list(roc_auc.values()))
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'macro_roc_auc': macro_roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    categories: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = True,
    cmap: str = 'Blues'
):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        categories: List of category names
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
        cmap: Color map for heatmap
    """
    plt.figure(figsize=(12, 10))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        xticklabels=categories,
        yticklabels=categories,
        cmap=cmap,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.show()


def plot_roc_curves(
    fpr: Dict[int, np.ndarray],
    tpr: Dict[int, np.ndarray],
    roc_auc: Dict[int, float],
    categories: List[str],
    save_path: str = "roc_curves.png"
):
    """
    Plot and save ROC curves.
    
    Args:
        fpr: False positive rates for each class
        tpr: True positive rates for each class
        roc_auc: AUC scores for each class
        categories: List of category names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for i, (cls, color) in enumerate(zip(categories, colors)):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f'{cls} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for All Classes', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add macro average AUC
    macro_auc = np.mean(list(roc_auc.values()))
    plt.text(0.95, 0.05, f'Macro Avg AUC = {macro_auc:.3f}',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.show()


def plot_classification_report(
    report: Dict[str, Any],
    categories: List[str],
    save_path: str = "classification_report.png"
):
    """
    Plot and save classification report as heatmap.
    
    Args:
        report: Classification report dictionary
        categories: List of category names
        save_path: Path to save plot
    """
    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    
    for cls in categories:
        row = [report[cls][metric] for metric in metrics]
        data.append(row)
    
    # Add support
    for i, cls in enumerate(categories):
        data[i].append(report[cls]['support'])
    
    metrics.append('support')
    
    # Create DataFrame
    df = pd.DataFrame(data, index=categories, columns=metrics)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df.iloc[:, :-1],  # Exclude support from heatmap
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Score'},
        vmin=0,
        vmax=1
    )
    
    plt.title('Classification Report (Per Class Metrics)', fontsize=16, pad=20)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Class', fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Add support as text
    for i, cls in enumerate(categories):
        plt.text(
            len(metrics) - 0.5,
            i + 0.5,
            f"Support: {int(df.loc[cls, 'support'])}",
            ha='left',
            va='center',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.show()


def save_evaluation_results(
    results: Dict[str, Any],
    categories: List[str],
    save_dir: str = "results/"
):
    """
    Save all evaluation results to files.
    
    Args:
        results: Evaluation results dictionary
        categories: List of category names
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save accuracy
    with open(os.path.join(save_dir, 'accuracy.txt'), 'w') as f:
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Macro ROC-AUC: {results['macro_roc_auc']:.4f}\n")
        f.write(f"Weighted F1-Score: {results['f1_score']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
    
    # Save classification report
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        results['confusion_matrix'],
        index=categories,
        columns=categories
    )
    cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
    
    # Save predictions and probabilities
    predictions_df = pd.DataFrame({
        'true_label': results['targets'],
        'predicted_label': results['predictions'],
        'true_class': [categories[i] for i in results['targets']],
        'predicted_class': [categories[i] for i in results['predictions']]
    })
    
    # Add probabilities for each class
    for i, cls in enumerate(categories):
        predictions_df[f'prob_{cls}'] = results['probabilities'][:, i]
    
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # Save ROC-AUC scores
    roc_auc_df = pd.DataFrame({
        'class': categories,
        'roc_auc': [results['roc_auc'][i] for i in range(len(categories))]
    })
    roc_auc_df.to_csv(os.path.join(save_dir, 'roc_auc_scores.csv'), index=False)
    
    # Generate plots
    plot_confusion_matrix(
        results['confusion_matrix'],
        categories,
        os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    plot_roc_curves(
        results['fpr'],
        results['tpr'],
        results['roc_auc'],
        categories,
        os.path.join(save_dir, 'roc_curves.png')
    )
    
    plot_classification_report(
        results['classification_report'],
        categories,
        os.path.join(save_dir, 'classification_report_heatmap.png')
    )
    
    print(f"Evaluation results saved to: {save_dir}")


def measure_inference_time(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Measure inference time of the model.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for inference
        device: Device to run inference on
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    
    # Warm-up
    print("Warming up...")
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break
    
    # Measure inference time
    print(f"Measuring inference time over {num_runs} runs...")
    total_time = 0.0
    total_images = 0
    
    with torch.no_grad():
        for run in range(num_runs):
            run_time = 0.0
            run_images = 0
            
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                # Sync GPU if using CUDA
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(inputs)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                batch_time = end_time - start_time
                batch_images = inputs.size(0)
                
                run_time += batch_time
                run_images += batch_images
            
            avg_time_per_image = (run_time / run_images) * 1000  # Convert to ms
            total_time += run_time
            total_images += run_images
            
            print(f"Run {run + 1}: {avg_time_per_image:.2f} ms/image")
    
    # Calculate average over all runs
    avg_time_per_image = (total_time / total_images) * 1000
    fps = 1000 / avg_time_per_image  # Frames per second
    
    return {
        'avg_inference_time_ms': avg_time_per_image,
        'fps': fps,
        'total_images': total_images,
        'total_time_seconds': total_time
    }


def main():
    """Main function for testing."""
    import argparse
    from src.model import EfficientNetLSTMModel
    from src.dataset import create_data_loaders
    from src.utils import load_config
    
    parser = argparse.ArgumentParser(description='Evaluate EfficientNet-LSTM model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default='data/EuroSAT_RGB',
                       help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    categories = config['dataset']['classes']
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders (test only)
    print("Creating data loaders...")
    _, _, test_loader = create_data_loaders(
        dataset_path=args.data_dir,
        categories=categories,
        image_size=tuple(config['dataset'].get('image_size', [128, 128])),
        batch_size=config['training']['batch_size'],
        split_ratios=config['dataset']['split_ratios'],
        random_seed=config['dataset']['random_seed'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Create model
    print("Loading model...")
    model = EfficientNetLSTMModel(
        num_classes=len(categories),
        dropout_rates=config['model']['dropout_rates'],
        lstm_units=config['model']['lstm_units'],
        dense_units=[config['model']['elm_hidden_units'], 128],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from: {args.model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device, categories)
    
    # Measure inference time
    print("\nMeasuring inference time...")
    timing_results = measure_inference_time(model, test_loader, device)
    
    # Save results
    print("\nSaving results...")
    save_evaluation_results(results, categories, args.save_dir)
    
    # Save timing results
    with open(os.path.join(args.save_dir, 'inference_timing.txt'), 'w') as f:
        f.write("Inference Timing Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Average inference time: {timing_results['avg_inference_time_ms']:.2f} ms/image\n")
        f.write(f"Frames per second: {timing_results['fps']:.2f} FPS\n")
        f.write(f"Total images processed: {timing_results['total_images']}\n")
        f.write(f"Total time: {timing_results['total_time_seconds']:.2f} seconds\n")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Macro ROC-AUC: {results['macro_roc_auc']:.4f}")
    print(f"Weighted F1-Score: {results['f1_score']:.4f}")
    print(f"Inference Time: {timing_results['avg_inference_time_ms']:.2f} ms/image")
    print(f"FPS: {timing_results['fps']:.2f}")
    print("=" * 50)
    
    # Print per-class accuracy
    print("\nPer-class accuracy from confusion matrix:")
    cm = results['confusion_matrix']
    for i, cls in enumerate(categories):
        class_acc = cm[i, i] / cm[i, :].sum() * 100
        print(f"{cls:25s}: {class_acc:.2f}%")
    
    print(f"\nAll results saved to: {args.save_dir}")


if __name__ == "__main__":
    import time
    main()