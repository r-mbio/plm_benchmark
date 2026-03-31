"""
EC Benchmark - Protein Language Models for EC Number Prediction
===============================================================

A comprehensive benchmark comparing protein language models (PLMs) for 
enzyme commission (EC) number prediction.

Modules:
    config: Configuration and paths
    data: Data loading utilities
    models: Neural network architectures
    metrics: Evaluation metrics
    train: Training script
    predict: Prediction script

Author: Rajesh Sathyamoorthy
"""

from .config import (
    EMBEDDINGS, EMBEDDING_DIMS, PLM_DISPLAY_NAMES,
    EC_LEVELS, THRESHOLDS, ARCHITECTURES, TRAIN_CONFIG
)
from .models import get_model, count_parameters
from .data import load_embeddings, load_labels, load_clusters, prepare_data
from .metrics import compute_metrics, compute_epoch_metrics

__version__ = '1.0.0'
__author__ = 'Rajesh Sathyamoorthy'
