"""
Training Script for EC Prediction
==================================
Train and evaluate models for EC number prediction from PLM embeddings.

Features:
- Multiple replicates for statistical robustness
- Early stopping with patience
- Mixed precision training (BF16/FP16)
- Comprehensive logging and metrics
- Cluster-based train/test splitting

Usage:
    # Train single model
    python train.py --plm esm2_650m --ec ec4 --threshold 50 --arch mlp
    
    # Run full benchmark
    python train.py --all

Author: Rajesh Sathyamoorthy
"""

import os
import sys
import json
import argparse
import time
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EMBEDDINGS, EMBEDDING_ID_KEYS, EMBEDDING_DIMS, LABEL_FILE, CLUSTERS,
    TRAIN_CONFIG, EC_LEVELS, THRESHOLDS, ARCHITECTURES, PLM_DISPLAY_NAMES,
    MODEL_DIR, RESULT_DIR, setup_dirs
)
from models import get_model, count_parameters
from data import load_embeddings, load_labels, load_clusters, prepare_data
from metrics import compute_metrics, compute_epoch_metrics


def set_global_seed(seed: int, deterministic: bool = True):
    """Set seeds and deterministic backend flags for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by cuBLAS for deterministic behavior on CUDA.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    """Ensure DataLoader worker RNG is deterministic across runs."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging to console and optionally to file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def train_epoch(model, loader, criterion, optimizer, device, use_amp, amp_dtype):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, X, y, criterion, device, use_amp, amp_dtype):
    """Evaluate model, return loss and predictions."""
    model.eval()
    
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        logits = model(X)
        loss = criterion(logits, y).item()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    
    return {'loss': loss, 'probs': probs, 'preds': preds}


def train_model(model, X_train, y_train, X_test, y_test, config, n_classes, seed, logger=None):
    """
    Train model with full logging and early stopping.
    
    Returns:
        Dictionary with final metrics and training history
    """
    device = next(model.parameters()).device
    
    use_cuda_device = str(device).startswith('cuda')
    requested_workers = int(config.get('num_workers', 0))
    if not use_cuda_device:
        # Multiprocessing dataloaders often fail in restricted/containerized CPU-only
        # environments; keep defaults portable.
        requested_workers = 0

    # Create data loader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=requested_workers,
        pin_memory=bool(config.get('pin_memory', True)) and use_cuda_device,
        persistent_workers=True if requested_workers > 0 else False,
        worker_init_fn=seed_worker if requested_workers > 0 else None,
        generator=data_gen,
    )
    
    # Test data as tensors
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    # AMP settings
    use_amp = config.get('use_amp', True) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if config.get('amp_dtype') == 'bfloat16' else torch.float16
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # Early stopping
    best_metric = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, amp_dtype)
        
        # Evaluate on test set
        val_eval = evaluate(model, X_test_t, y_test_t, criterion, device, use_amp, amp_dtype)
        val_metrics = compute_epoch_metrics(y_test, val_eval['preds'], val_eval['probs'])
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_eval['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Check for improvement
        current_metric = val_metrics['accuracy']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        if logger and (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}, acc={val_metrics['accuracy']:.4f}, f1={val_metrics['f1']:.4f}")
        
        # Early stopping
        if patience_counter >= config['early_stop_patience']:
            if logger:
                logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        y_prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
    
    final_metrics = compute_metrics(y_test, y_pred, y_prob, n_classes)
    final_metrics['best_epoch'] = best_epoch
    final_metrics['total_epochs'] = epoch + 1
    final_metrics['history'] = history
    
    return final_metrics


def run_experiment(plm, ec, threshold, arch, replicate, config, device, logger=None):
    """Run a single experiment."""
    seed = config['seeds'][replicate - 1]
    
    if logger:
        logger.info(f"  PLM={plm}, EC={ec}, Threshold={threshold}%, Arch={arch}, Rep={replicate}")
    
    t0 = time.time()
    
    # Set seeds/backend determinism
    set_global_seed(seed, deterministic=config.get('deterministic', True))
    
    # Load data
    embeddings, dim = load_embeddings(EMBEDDINGS[plm], EMBEDDING_ID_KEYS[plm])
    labels, class_names = load_labels(LABEL_FILE, ec)
    clusters = load_clusters(CLUSTERS[threshold])
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, n_classes, label_encoder, train_ids, test_ids = prepare_data(
        embeddings, labels, clusters,
        min_samples=config['min_samples_per_class'],
        train_ratio=config['train_ratio'],
        seed=seed
    )
    
    if logger:
        logger.info(f"    Data: train={len(X_train)}, test={len(X_test)}, classes={n_classes}")
    
    # Create model
    model = get_model(arch, dim, n_classes, config['dropout'], device)
    n_params = count_parameters(model)
    
    # Train
    metrics = train_model(model, X_train, y_train, X_test, y_test, config, n_classes, seed, logger)
    
    elapsed = time.time() - t0
    
    # Persist model checkpoint for downstream external validation.
    ckpt_dir = MODEL_DIR / plm
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{ec}_{threshold}pct_{arch}_rep{replicate}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Save minimal label-mapping metadata for audit/reproducibility.
    ckpt_meta_path = ckpt_dir / f"{ec}_{threshold}pct_{arch}_rep{replicate}.meta.json"
    ckpt_meta = {
        "plm": plm,
        "ec_level": ec,
        "threshold": threshold,
        "architecture": arch,
        "replicate": replicate,
        "seed": seed,
        "num_classes": int(n_classes),
        "label_encoder_classes": [int(x) for x in label_encoder.classes_.tolist()],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    with open(ckpt_meta_path, "w", encoding="utf-8") as f:
        json.dump(ckpt_meta, f, indent=2)

    # Compile results
    result = {
        'plm': plm,
        'plm_display': PLM_DISPLAY_NAMES[plm],
        'ec_level': ec,
        'threshold': threshold,
        'architecture': arch,
        'replicate': replicate,
        'seed': seed,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'num_classes': n_classes,
        'embedding_dim': dim,
        'model_params': n_params,
        'checkpoint_path': str(ckpt_path),
        'checkpoint_meta_path': str(ckpt_meta_path),
        'time_seconds': elapsed,
        **metrics
    }
    
    if logger:
        logger.info(f"    Result: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}, time={elapsed:.1f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='EC Prediction Benchmark')
    parser.add_argument('--plm', type=str, choices=list(EMBEDDINGS.keys()))
    parser.add_argument('--ec', type=str, choices=['ec1', 'ec2', 'ec3', 'ec4'])
    parser.add_argument('--threshold', type=int, choices=[30, 50, 70, 90])
    parser.add_argument('--arch', type=str, choices=ARCHITECTURES)
    parser.add_argument('--replicate', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--all', action='store_true', help='Run full benchmark')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help='Force deterministic training behavior')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                        help='Allow non-deterministic kernels for speed')
    parser.set_defaults(deterministic=TRAIN_CONFIG.get('deterministic', True))
    args = parser.parse_args()
    
    setup_dirs()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = RESULT_DIR / f'train_{timestamp}.log'
    logger = setup_logging(log_file)

    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    run_config = TRAIN_CONFIG.copy()
    run_config['deterministic'] = bool(args.deterministic)
    logger.info(f"Deterministic mode: {run_config['deterministic']}")
    
    if args.all:
        # Run full benchmark
        results = []
        for plm in EMBEDDINGS.keys():
            for ec in EC_LEVELS:
                for threshold in THRESHOLDS:
                    for arch in ARCHITECTURES:
                        for rep in range(1, run_config['num_replicates'] + 1):
                            result = run_experiment(plm, ec, threshold, arch, rep, run_config, device, logger)
                            results.append(result)

        # Save results
        import pandas as pd
        df = pd.DataFrame(results)
        out_ts = RESULT_DIR / f'benchmark_results_{timestamp}.csv'
        out_latest = RESULT_DIR / 'benchmark_results_latest.csv'
        df.to_csv(out_ts, index=False)
        df.to_csv(out_latest, index=False)
        logger.info(f"Saved {len(results)} results")
        logger.info(f"Saved results to: {out_ts}")
        logger.info(f"Updated latest results: {out_latest}")
    else:
        # Run single experiment
        if not all([args.plm, args.ec, args.threshold, args.arch]):
            parser.error("Specify --plm, --ec, --threshold, --arch or use --all")
        
        result = run_experiment(args.plm, args.ec, args.threshold, args.arch, 
                               args.replicate, run_config, device, logger)
        print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
