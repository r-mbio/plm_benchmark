#!/usr/bin/env python3
"""
Save Best Models for EC Prediction
==================================
Trains and saves the best models for each sequence identity threshold (30%, 50%, 70%, 90%)
using ESM2-650M embeddings, plus a binary enzyme classifier.

Models saved:
- ec4_classifier_30pct.pt  (hardest - true generalization)
- ec4_classifier_50pct.pt  (standard benchmark)
- ec4_classifier_70pct.pt  
- ec4_classifier_90pct.pt  (easiest)
- binary_classifier.pt     (enzyme vs non-enzyme)

Usage:
    python scripts/save_best_models.py

Author: Rajesh Sathyamoorthy
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# =============================================================================

# Data paths (update these to match your VM)
DATA_DIR = Path("/data/rsathyamo/ec/data")
EMBEDDINGS_PATH = Path("/data/rsathyamo/ec_pred/data/prebuilt/uniprot_esm2_t33_650M.h5")
LABELS_PATH = DATA_DIR / "labels/ec_labels.npz"
CLUSTERS_DIR = DATA_DIR / "clusters"

# Binary classifier data (if available)
BINARY_DATA_PATH = DATA_DIR / "binary_classifier/balanced_dataset.tsv"
BINARY_EMB_PATH = DATA_DIR / "binary_classifier/embeddings_esm2_650m.h5"

# Output directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Training settings
BATCH_SIZE = 2048
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Model output: {MODEL_DIR}")

# =============================================================================
# MODEL ARCHITECTURE (MLP - best performing)
# =============================================================================

class MLPClassifier(nn.Module):
    """Simple MLP classifier - best architecture from benchmark."""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings(h5_path):
    """Load embeddings from HDF5 file."""
    print(f"Loading embeddings from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        
        # Try different key combinations
        if 'embeddings' in keys:
            X = f['embeddings'][:]
            if 'ids' in keys:
                ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['ids'][:]]
            elif 'uniprot_ids' in keys:
                ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['uniprot_ids'][:]]
            else:
                ids = [str(i) for i in range(len(X))]
        else:
            # Per-protein format
            ids = list(keys)
            X = np.array([f[pid][:].mean(axis=0) if f[pid][:].ndim == 2 else f[pid][:] for pid in ids])
    
    print(f"  Loaded {len(ids)} proteins, embedding dim: {X.shape[1]}")
    return {pid: X[i] for i, pid in enumerate(ids)}, X.shape[1]


def load_labels(npz_path):
    """Load EC labels from NPZ file."""
    print(f"Loading labels from {npz_path}...")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Get protein IDs
    if 'uniprot_ids' in data:
        ids = data['uniprot_ids']
    elif 'ids' in data:
        ids = data['ids']
    else:
        raise KeyError("No ID key found in labels file")
    
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
    
    # Get EC4 multilabel matrix - handle different formats
    ec4_multilabel = data['ec4_multilabel']
    
    # If it's a sparse matrix, convert to dense
    if hasattr(ec4_multilabel, 'toarray'):
        ec4_multilabel = ec4_multilabel.toarray()
    # If it's a 0-d array containing an object (e.g., sparse matrix)
    elif ec4_multilabel.ndim == 0:
        obj = ec4_multilabel.item()
        if hasattr(obj, 'toarray'):
            ec4_multilabel = obj.toarray()
        else:
            ec4_multilabel = np.array(obj)
    # Otherwise it's already a dense array - just ensure it's numpy
    else:
        ec4_multilabel = np.asarray(ec4_multilabel)
    
    # Get class names if available
    if 'ec4_classes' in data:
        ec_classes = [str(x) for x in data['ec4_classes']]
    else:
        ec_classes = [str(i) for i in range(ec4_multilabel.shape[1])]
    
    print(f"  Loaded {len(ids)} proteins, {len(ec_classes)} EC classes")
    return ids, ec4_multilabel, ec_classes


def load_clusters(cluster_path):
    """Load MMseqs2 clusters from TSV file."""
    print(f"Loading clusters from {cluster_path}...")
    
    clusters = defaultdict(list)
    with open(cluster_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters[rep].append(member)
    
    print(f"  Loaded {len(clusters)} clusters")
    return clusters


def create_train_test_split(clusters, train_ratio=0.8, seed=42):
    """Split clusters into train/test sets."""
    np.random.seed(seed)
    
    cluster_ids = list(clusters.keys())
    np.random.shuffle(cluster_ids)
    
    n_train = int(len(cluster_ids) * train_ratio)
    train_clusters = set(cluster_ids[:n_train])
    test_clusters = set(cluster_ids[n_train:])
    
    train_ids = set()
    test_ids = set()
    
    for cid in train_clusters:
        train_ids.update(clusters[cid])
    for cid in test_clusters:
        test_ids.update(clusters[cid])
    
    return train_ids, test_ids

# =============================================================================
# TRAINING
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, num_classes, input_dim):
    """Train MLP classifier with early stopping."""
    
    model = MLPClassifier(input_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_acc

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SAVING BEST MODELS FOR EC PREDICTION")
    print("=" * 70)
    
    # Load embeddings
    emb_dict, emb_dim = load_embeddings(EMBEDDINGS_PATH)
    
    # Load labels
    label_ids, ec4_multilabel, ec_classes = load_labels(LABELS_PATH)
    
    # Convert multilabel to single label (argmax)
    ec_single = ec4_multilabel.argmax(axis=1)
    has_ec = ec4_multilabel.sum(axis=1) > 0
    
    # Create ID to label mapping
    id_to_label = {}
    for i, pid in enumerate(label_ids):
        if has_ec[i]:
            id_to_label[pid] = ec_single[i]
    
    print(f"\nProteins with EC annotations: {len(id_to_label)}")
    
    # Train models for each threshold
    thresholds = [30, 50, 70, 90]
    
    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"TRAINING EC4 CLASSIFIER AT {threshold}% SEQUENCE IDENTITY")
        print(f"{'='*70}")
        
        cluster_path = CLUSTERS_DIR / f"clusters_{threshold}pct.tsv"
        if not cluster_path.exists():
            print(f"  WARNING: Cluster file not found: {cluster_path}")
            continue
        
        # Load clusters and create split
        clusters = load_clusters(cluster_path)
        train_ids, test_ids = create_train_test_split(clusters, train_ratio=0.8, seed=42)
        
        # Prepare data
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        for pid in emb_dict:
            if pid in id_to_label:
                if pid in train_ids:
                    X_train_list.append(emb_dict[pid])
                    y_train_list.append(id_to_label[pid])
                elif pid in test_ids:
                    X_test_list.append(emb_dict[pid])
                    y_test_list.append(id_to_label[pid])
        
        X_train = np.array(X_train_list)
        y_train_raw = np.array(y_train_list)
        X_test = np.array(X_test_list)
        y_test_raw = np.array(y_test_list)
        
        # Remap classes to contiguous indices
        unique_classes = np.unique(np.concatenate([y_train_raw, y_test_raw]))
        class_map = {c: i for i, c in enumerate(unique_classes)}
        inv_class_map = {i: c for c, i in class_map.items()}
        
        y_train = np.array([class_map[y] for y in y_train_raw])
        y_test = np.array([class_map[y] for y in y_test_raw])
        
        num_classes = len(class_map)
        
        print(f"  Train: {len(X_train)} proteins")
        print(f"  Test: {len(X_test)} proteins")
        print(f"  Classes: {num_classes}")
        
        # Train model
        model, val_acc = train_model(X_train, y_train, X_test, y_test, num_classes, emb_dim)
        
        print(f"  Best validation accuracy: {val_acc*100:.2f}%")
        
        # Save model
        model_path = MODEL_DIR / f"ec4_classifier_{threshold}pct.pt"
        
        # Create class mapping for inference
        ec_class_names = {i: ec_classes[inv_class_map[i]] for i in range(num_classes)}
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'input_dim': emb_dim,
            'num_classes': num_classes,
            'class_map': class_map,
            'inv_class_map': inv_class_map,
            'ec_class_names': ec_class_names,
            'threshold': threshold,
            'architecture': 'mlp',
            'hidden_dims': [512, 256],
            'val_accuracy': val_acc,
            'plm': 'esm2_650m',
        }
        
        torch.save(save_dict, model_path)
        print(f"  Saved: {model_path}")
        
        # Also save class names as JSON
        json_path = MODEL_DIR / f"ec4_classes_{threshold}pct.json"
        with open(json_path, 'w') as f:
            json.dump(ec_class_names, f, indent=2)
        print(f"  Saved: {json_path}")
    
    # ==========================================================================
    # BINARY CLASSIFIER (enzyme vs non-enzyme)
    # ==========================================================================
    print(f"\n{'='*70}")
    print("TRAINING BINARY CLASSIFIER (Enzyme vs Non-Enzyme)")
    print(f"{'='*70}")
    
    if BINARY_DATA_PATH.exists() and BINARY_EMB_PATH.exists():
        import pandas as pd
        
        # Load binary dataset
        binary_df = pd.read_csv(BINARY_DATA_PATH, sep='\t')
        print(f"  Binary dataset: {len(binary_df)} proteins")
        
        # Load embeddings
        binary_emb, _ = load_embeddings(BINARY_EMB_PATH)
        
        # Prepare data
        X_list, y_list = [], []
        for _, row in binary_df.iterrows():
            pid = str(row['uniprot_id']) if 'uniprot_id' in row else str(row['id'])
            if pid in binary_emb:
                X_list.append(binary_emb[pid])
                y_list.append(int(row['label']))
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"  Train: {len(X_train)} proteins")
        print(f"  Test: {len(X_test)} proteins")
        
        # Train
        model, val_acc = train_model(X_train, y_train, X_test, y_test, num_classes=2, input_dim=emb_dim)
        
        print(f"  Best validation accuracy: {val_acc*100:.2f}%")
        
        # Save
        model_path = MODEL_DIR / "binary_classifier.pt"
        save_dict = {
            'model_state_dict': model.state_dict(),
            'input_dim': emb_dim,
            'num_classes': 2,
            'class_names': {0: 'non-enzyme', 1: 'enzyme'},
            'architecture': 'mlp',
            'hidden_dims': [512, 256],
            'val_accuracy': val_acc,
            'plm': 'esm2_650m',
        }
        torch.save(save_dict, model_path)
        print(f"  Saved: {model_path}")
    else:
        print(f"  WARNING: Binary classifier data not found")
        print(f"    Expected: {BINARY_DATA_PATH}")
        print(f"    Expected: {BINARY_EMB_PATH}")
    
    print(f"\n{'='*70}")
    print("DONE! Models saved to:", MODEL_DIR)
    print(f"{'='*70}")
    
    # List saved models
    print("\nSaved models:")
    for f in sorted(MODEL_DIR.glob("*.pt")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
