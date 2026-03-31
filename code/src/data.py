"""
Data Loading Utilities
======================
Robust data loading with validation for EC benchmark.

Handles:
- HDF5 embedding files with different key names
- Byte string decoding
- Cluster-based train/test splitting (prevents data leakage)
- Multilabel to single-label conversion for EC1-4

Author: Rajesh Sathyamoorthy
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.preprocessing import LabelEncoder


def decode_ids(ids: np.ndarray) -> List[str]:
    """Decode byte strings to regular strings."""
    if len(ids) == 0:
        return []
    
    sample = ids[0]
    
    if isinstance(sample, bytes):
        return [x.decode('utf-8') for x in ids]
    elif isinstance(sample, np.bytes_):
        return [x.decode('utf-8') for x in ids]
    elif hasattr(sample, 'decode'):
        return [x.decode('utf-8') for x in ids]
    elif isinstance(sample, str):
        return list(ids)
    elif isinstance(sample, np.str_):
        return [str(x) for x in ids]
    else:
        return [str(x) for x in ids]


def load_embeddings(filepath: Path, id_key: str = None) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load protein embeddings from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        id_key: Specific key for IDs (if None, auto-detect)
    
    Returns:
        Tuple of (dict mapping protein_id -> embedding, embedding_dim)
    """
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())

        # Standard table format: separate ID and embedding arrays.
        if ('embeddings' in keys) or any(k in keys for k in ['uniprot_ids', 'ids', 'protein_ids', 'accessions', 'id']):
            if id_key and id_key in keys:
                pass
            else:
                id_key = None
                for candidate in ['uniprot_ids', 'ids', 'protein_ids', 'accessions', 'id']:
                    if candidate in keys:
                        id_key = candidate
                        break

            if id_key is None:
                raise KeyError(f"No ID key found in {filepath}. Available: {keys}")

            emb_key = 'embeddings' if 'embeddings' in keys else [k for k in keys if k != id_key][0]
            ids = decode_ids(f[id_key][:])
            embeddings = f[emb_key][:]
            return dict(zip(ids, embeddings)), int(embeddings.shape[1])

        # Per-protein format: each H5 key is a protein accession dataset.
        # Example: esm2_8m.h5 / esm2_35m.h5.
        embeddings: Dict[str, np.ndarray] = {}
        emb_dim = None
        for pid in keys:
            arr = f[pid][:]
            if arr.ndim == 2:
                # Mean-pool token embeddings to one vector per protein.
                vec = arr.mean(axis=0)
            elif arr.ndim == 1:
                vec = arr
            else:
                vec = np.asarray(arr).reshape(-1)

            vec = np.asarray(vec, dtype=np.float32)
            if emb_dim is None:
                emb_dim = int(vec.shape[0])
            elif int(vec.shape[0]) != emb_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions in {filepath}: "
                    f"expected {emb_dim}, got {vec.shape[0]} for {pid}"
                )
            embeddings[str(pid)] = vec

    if not embeddings or emb_dim is None:
        raise ValueError(f"No embeddings loaded from {filepath}")
    return embeddings, emb_dim


def load_labels(filepath: Path, level: str = 'ec4') -> Tuple[Dict[str, int], List[str]]:
    """
    Load EC labels from NPZ file (single-label format).
    
    Args:
        filepath: Path to NPZ file
        level: EC level ('ec1', 'ec2', 'ec3', 'ec4')
    
    Returns:
        Tuple of (dict mapping protein_id -> label_index, list of class names)
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Find ID key
    id_key = None
    for key in ['uniprot_ids', 'ids', 'protein_ids']:
        if key in data.files:
            id_key = key
            break
    
    if id_key is None:
        raise KeyError(f"No ID key found. Available: {data.files}")
    
    ids = decode_ids(data[id_key])
    ec4_classes = list(data['ec4_classes'])
    ec4_multilabel = data['ec4_multilabel']
    if hasattr(ec4_multilabel, 'toarray'):
        ec4_multilabel = ec4_multilabel.toarray()

    # Keep only proteins with at least one EC label.
    # This avoids assigning all-zero rows to class 0 via argmax.
    has_label = ec4_multilabel.sum(axis=1) > 0
    ids = [uid for uid, keep in zip(ids, has_label) if keep]
    ec4_multilabel = ec4_multilabel[has_label]

    if level == 'ec4':
        # Multi-label rows are deterministically projected to the smallest
        # positive class index (stable across runs and environments).
        labels = np.array([int(np.flatnonzero(row > 0)[0]) for row in ec4_multilabel], dtype=np.int64)
        return dict(zip(ids, labels)), ec4_classes
    
    elif level in ['ec1', 'ec2', 'ec3']:
        n_parts = {'ec1': 1, 'ec2': 2, 'ec3': 3}[level]
        
        # Truncate EC4 classes to desired level
        truncated = ['.'.join(ec.split('.')[:n_parts]) for ec in ec4_classes]
        unique_truncated = sorted(set(truncated))
        trunc_to_idx = {t: i for i, t in enumerate(unique_truncated)}
        ec4_to_trunc = [trunc_to_idx[t] for t in truncated]
        
        # Map each protein to truncated EC
        labels = []
        for i in range(len(ids)):
            ec4_indices = np.flatnonzero(ec4_multilabel[i] > 0)
            trunc_idx = ec4_to_trunc[int(ec4_indices[0])]
            labels.append(trunc_idx)
        
        return dict(zip(ids, labels)), unique_truncated
    
    else:
        raise ValueError(f"Unknown level: {level}. Use 'ec1', 'ec2', 'ec3', or 'ec4'")


def load_clusters(filepath: Path) -> Dict[str, str]:
    """
    Load cluster assignments from TSV file.
    
    Format: representative<TAB>member
    
    Args:
        filepath: Path to TSV file
    
    Returns:
        Dict mapping member_id -> representative_id (cluster)
    """
    clusters = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                clusters[member] = representative
    return clusters


def prepare_data(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, int],
    clusters: Dict[str, str],
    min_samples: int = 10,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, LabelEncoder, List[str], List[str]]:
    """
    Prepare train/test data with cluster-based splitting.
    
    This ensures no sequence similarity leakage between train and test sets.
    
    Args:
        embeddings: Dict mapping protein_id -> embedding
        labels: Dict mapping protein_id -> label_index
        clusters: Dict mapping protein_id -> cluster_id
        min_samples: Minimum samples per class in training
        train_ratio: Fraction of clusters for training
        seed: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, num_classes, label_encoder, train_ids, test_ids)
    """
    np.random.seed(seed)
    
    # Find common proteins
    common = set(embeddings.keys()) & set(labels.keys()) & set(clusters.keys())
    common = sorted(list(common))
    
    if len(common) == 0:
        raise ValueError("No common proteins found across embeddings, labels, and clusters!")
    
    # Group proteins by cluster
    cluster_to_proteins = {}
    for uid in common:
        cluster_id = clusters[uid]
        cluster_to_proteins.setdefault(cluster_id, []).append(uid)
    
    # Split clusters into train/test
    cluster_ids = list(cluster_to_proteins.keys())
    np.random.shuffle(cluster_ids)
    n_train = int(len(cluster_ids) * train_ratio)
    train_clusters = set(cluster_ids[:n_train])
    
    train_ids = [uid for uid in common if clusters[uid] in train_clusters]
    test_ids = [uid for uid in common if clusters[uid] not in train_clusters]
    
    # Get raw labels
    y_train_raw = [labels[uid] for uid in train_ids]
    y_test_raw = [labels[uid] for uid in test_ids]
    
    # Count classes in training
    train_label_counts = {}
    for lbl in y_train_raw:
        train_label_counts[lbl] = train_label_counts.get(lbl, 0) + 1
    
    # Filter to classes with min_samples
    valid_classes = {lbl for lbl, cnt in train_label_counts.items() if cnt >= min_samples}
    
    # Filter train and test
    train_mask = [lbl in valid_classes for lbl in y_train_raw]
    test_mask = [lbl in valid_classes for lbl in y_test_raw]
    
    train_ids = [uid for uid, m in zip(train_ids, train_mask) if m]
    test_ids = [uid for uid, m in zip(test_ids, test_mask) if m]
    
    y_train_raw = [labels[uid] for uid in train_ids]
    y_test_raw = [labels[uid] for uid in test_ids]
    
    # Encode labels (re-index to 0..n_classes-1)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)
    
    # Build X arrays
    X_train = np.array([embeddings[uid] for uid in train_ids], dtype=np.float32)
    X_test = np.array([embeddings[uid] for uid in test_ids], dtype=np.float32)
    
    return X_train, X_test, y_train, y_test, num_classes, label_encoder, train_ids, test_ids
