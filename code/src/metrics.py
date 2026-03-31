"""
Evaluation Metrics
==================
Comprehensive metrics for EC prediction evaluation.

Includes:
- Standard classification metrics (accuracy, precision, recall, F1)
- Multi-class metrics (macro, micro, weighted averaging)
- Matthews Correlation Coefficient (MCC)
- ROC-AUC scores
- Per-class performance

Author: Rajesh Sathyamoorthy
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, log_loss,
    confusion_matrix, classification_report
)


def compute_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Compute metrics for a single epoch (lightweight version).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
    
    Returns:
        Dict with accuracy, precision, recall, f1
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_prob: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for final evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (n_samples x n_classes)
        n_classes: Number of classes
    
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Top-k accuracy
    if y_prob is not None:
        metrics['top3_accuracy'] = top_k_accuracy(y_true, y_prob, k=3)
        metrics['top5_accuracy'] = top_k_accuracy(y_true, y_prob, k=5)
        metrics['log_loss'] = log_loss(y_true, y_prob, labels=list(range(n_classes)))
    
    # Precision, Recall, F1 (multiple averaging methods)
    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    # MCC and Cohen's Kappa
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # ROC-AUC (if probabilities available)
    if y_prob is not None and n_classes > 2:
        try:
            # One-vs-rest ROC-AUC
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i, label in enumerate(y_true):
                y_true_onehot[i, label] = 1
            
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true_onehot, y_prob, average='macro', multi_class='ovr'
            )
            metrics['roc_auc_weighted'] = roc_auc_score(
                y_true_onehot, y_prob, average='weighted', multi_class='ovr'
            )
        except ValueError:
            # Some classes may not be present in test set
            metrics['roc_auc_macro'] = np.nan
            metrics['roc_auc_weighted'] = np.nan
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Class support (number of samples per class)
    unique, counts = np.unique(y_true, return_counts=True)
    support = np.zeros(n_classes, dtype=int)
    for u, c in zip(unique, counts):
        support[u] = c
    
    metrics['per_class_precision'] = per_class_precision.tolist()
    metrics['per_class_recall'] = per_class_recall.tolist()
    metrics['per_class_f1'] = per_class_f1.tolist()
    metrics['per_class_support'] = support.tolist()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy score
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)


def format_metrics_table(metrics: Dict[str, Any], precision: int = 4) -> str:
    """
    Format metrics as a readable table.
    
    Args:
        metrics: Dict of metric name -> value
        precision: Decimal places for floating point values
    
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 50)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 50)
    
    # Main metrics
    main_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'mcc']
    lines.append("\nMain Metrics:")
    lines.append("-" * 30)
    for m in main_metrics:
        if m in metrics:
            lines.append(f"  {m:<25}: {metrics[m]:.{precision}f}")
    
    # Detailed metrics
    lines.append("\nDetailed Metrics:")
    lines.append("-" * 30)
    for key, value in sorted(metrics.items()):
        if key not in main_metrics and not key.startswith('per_class') and key != 'confusion_matrix':
            if isinstance(value, float):
                lines.append(f"  {key:<25}: {value:.{precision}f}")
            else:
                lines.append(f"  {key:<25}: {value}")
    
    return "\n".join(lines)
