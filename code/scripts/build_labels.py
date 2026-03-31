#!/usr/bin/env python3
"""
Build Clean Label File for EC Benchmark
========================================

Creates a comprehensive label file with:
- EC1, EC2, EC3, EC4 (hierarchical)
- Multi-label support
- Only proteins that have embeddings AND clusters

Usage:
    cd /data/rsathyamo/ec
    python scripts/build_labels.py
"""

import numpy as np
import h5py
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# =============================================================================
# Configuration - Update these paths as needed
# =============================================================================

# Project root
PROJECT_ROOT = Path("/data/rsathyamo/ec")
DATA_DIR = PROJECT_ROOT / "data"

# Input files - try multiple locations
OLD_LABELS_PATHS = [
    "/data/rsathyamo/esm2_structure_benchmark/data/labels/labels.npz",
    "/data/rsathyamo/ec_benchmark/data/labels.npz",
    "/data/rsathyamo/ec_pred/data/labels.npz",
]

# Embedding files (to get protein IDs)
EMBEDDING_FILE = DATA_DIR / "embeddings" / "esm2_650m.h5"

# Cluster file
CLUSTER_FILE = DATA_DIR / "clusters" / "clusters_30pct.tsv"

# Output
OUTPUT_FILE = DATA_DIR / "labels" / "ec_labels.npz"


def find_file(paths):
    """Find the first existing file from a list of paths."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def detect_h5_keys(filepath):
    """Detect the key names in an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        print(f"  Available keys: {keys}")
        
        # Find ID key
        id_key = None
        for k in ['uniprot_ids', 'ids', 'protein_ids', 'accessions', 'names']:
            if k in keys:
                id_key = k
                break
        
        # Find embedding key
        emb_key = None
        for k in ['embeddings', 'features', 'representations', 'data']:
            if k in keys:
                emb_key = k
                break
        
        return id_key, emb_key, keys


def load_protein_ids_from_embeddings():
    """Get protein IDs that have embeddings (our working set)."""
    print(f"Loading protein IDs from {EMBEDDING_FILE}...")
    
    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(f"Embedding file not found: {EMBEDDING_FILE}")
    
    id_key, emb_key, keys = detect_h5_keys(EMBEDDING_FILE)
    
    if id_key is None:
        raise KeyError(f"Could not find ID key in {EMBEDDING_FILE}. Keys: {keys}")
    
    with h5py.File(EMBEDDING_FILE, 'r') as f:
        ids = f[id_key][:]
        ids = [x.decode() if isinstance(x, bytes) else x for x in ids]
    
    print(f"  Found {len(ids)} proteins with embeddings")
    return set(ids)


def load_cluster_proteins():
    """Get protein IDs that have cluster assignments."""
    print(f"Loading clustered proteins from {CLUSTER_FILE}...")
    
    if not CLUSTER_FILE.exists():
        raise FileNotFoundError(f"Cluster file not found: {CLUSTER_FILE}")
    
    proteins = set()
    with open(CLUSTER_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                proteins.add(parts[1])  # member
    
    print(f"  Found {len(proteins)} proteins in clusters")
    return proteins


def load_old_labels():
    """Load existing EC4 labels from various possible locations."""
    print("Loading existing labels...")
    
    label_file = find_file(OLD_LABELS_PATHS)
    
    if label_file is None:
        print(f"  WARNING: Could not find labels file in any of:")
        for p in OLD_LABELS_PATHS:
            print(f"    - {p}")
        return None
    
    print(f"  Found labels at: {label_file}")
    
    data = np.load(label_file, allow_pickle=True)
    
    # Find ID key
    id_key = None
    for k in ['uniprot_ids', 'ids', 'protein_ids']:
        if k in data.files:
            id_key = k
            break
    
    if id_key is None:
        raise KeyError(f"Could not find ID key. Available: {data.files}")
    
    uniprot_ids = list(data[id_key])
    if len(uniprot_ids) > 0 and isinstance(uniprot_ids[0], bytes):
        uniprot_ids = [x.decode() for x in uniprot_ids]
    
    # Find EC4 classes
    ec4_classes = None
    for k in ['ec4_classes', 'ec_classes', 'classes']:
        if k in data.files:
            ec4_classes = list(data[k])
            break
    
    # Find EC4 multilabel matrix
    ec4_multilabel = None
    for k in ['ec4_multilabel', 'ec4', 'multilabel', 'labels']:
        if k in data.files:
            ec4_multilabel = data[k]
            break
    
    print(f"  Loaded {len(uniprot_ids)} proteins")
    if ec4_classes:
        print(f"  EC4 classes: {len(ec4_classes)}")
    
    return {
        'uniprot_ids': uniprot_ids,
        'ec4_classes': ec4_classes,
        'ec4_multilabel': ec4_multilabel,
    }


def derive_hierarchical_ec(ec4_classes, ec4_multilabel):
    """
    Derive EC1, EC2, EC3 from EC4.
    
    EC4: 1.2.3.4 -> EC3: 1.2.3 -> EC2: 1.2 -> EC1: 1
    """
    print("Deriving hierarchical EC labels...")
    
    n_proteins = ec4_multilabel.shape[0]
    results = {}
    
    for level, n_parts in [('ec1', 1), ('ec2', 2), ('ec3', 3)]:
        # Map EC4 index to truncated EC string
        ec4_to_truncated = {}
        for i, ec4 in enumerate(ec4_classes):
            parts = ec4.split('.')
            truncated = '.'.join(parts[:n_parts])
            ec4_to_truncated[i] = truncated
        
        # Get unique truncated classes
        unique_truncated = sorted(set(ec4_to_truncated.values()))
        truncated_to_idx = {ec: i for i, ec in enumerate(unique_truncated)}
        
        # Build multilabel matrix
        multilabel = np.zeros((n_proteins, len(unique_truncated)), dtype=np.int8)
        
        for protein_idx in range(n_proteins):
            ec4_indices = np.where(ec4_multilabel[protein_idx] > 0)[0]
            for ec4_idx in ec4_indices:
                truncated = ec4_to_truncated[ec4_idx]
                truncated_idx = truncated_to_idx[truncated]
                multilabel[protein_idx, truncated_idx] = 1
        
        # For single-label, take the first (most common) class
        single_label = np.argmax(multilabel, axis=1)
        
        results[level] = {
            'classes': unique_truncated,
            'multilabel': multilabel,
            'single': single_label,
        }
        
        # Count multi-label proteins
        multi_count = np.sum(np.sum(multilabel, axis=1) > 1)
        print(f"  {level.upper()}: {len(unique_truncated)} classes, {multi_count} multi-label proteins")
    
    return results


def main():
    print("=" * 60)
    print("Building Clean Label File")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load protein IDs from embeddings
    try:
        embedding_proteins = load_protein_ids_from_embeddings()
    except Exception as e:
        print(f"ERROR loading embeddings: {e}")
        return 1
    
    # Load cluster proteins
    try:
        cluster_proteins = load_cluster_proteins()
    except Exception as e:
        print(f"ERROR loading clusters: {e}")
        return 1
    
    # Working set: proteins with both embeddings and clusters
    working_set = embedding_proteins & cluster_proteins
    print(f"\nWorking set: {len(working_set)} proteins")
    
    # Load old labels
    old_labels = load_old_labels()
    
    if old_labels is None:
        print("\nERROR: Could not load existing labels.")
        print("Please ensure one of these files exists:")
        for p in OLD_LABELS_PATHS:
            print(f"  - {p}")
        return 1
    
    # Filter to working set
    print("Filtering to working set...")
    
    old_ids = old_labels['uniprot_ids']
    old_id_to_idx = {uid: i for i, uid in enumerate(old_ids)}
    
    # Find proteins in both working set and old labels
    common = working_set & set(old_ids)
    print(f"  {len(common)} proteins in working set (out of {len(old_ids)})")
    
    if len(common) == 0:
        print("\nERROR: No common proteins found!")
        print(f"  Working set sample: {list(working_set)[:5]}")
        print(f"  Old labels sample: {old_ids[:5]}")
        return 1
    
    # Build filtered arrays
    common_list = sorted(list(common))
    common_indices = [old_id_to_idx[uid] for uid in common_list]
    
    ec4_multilabel = old_labels['ec4_multilabel'][common_indices]
    ec4_classes = old_labels['ec4_classes']
    
    # Derive hierarchical labels
    hierarchical = derive_hierarchical_ec(ec4_classes, ec4_multilabel)
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    
    np.savez_compressed(
        OUTPUT_FILE,
        # IDs
        uniprot_ids=np.array(common_list),
        
        # EC4
        ec4=hierarchical.get('ec4', {}).get('single', np.argmax(ec4_multilabel, axis=1)),
        ec4_classes=np.array(ec4_classes),
        ec4_multilabel=ec4_multilabel,
        
        # EC3
        ec3=hierarchical['ec3']['single'],
        ec3_classes=np.array(hierarchical['ec3']['classes']),
        ec3_multilabel=hierarchical['ec3']['multilabel'],
        
        # EC2
        ec2=hierarchical['ec2']['single'],
        ec2_classes=np.array(hierarchical['ec2']['classes']),
        ec2_multilabel=hierarchical['ec2']['multilabel'],
        
        # EC1
        ec1=hierarchical['ec1']['single'],
        ec1_classes=np.array(hierarchical['ec1']['classes']),
        ec1_multilabel=hierarchical['ec1']['multilabel'],
    )
    
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Proteins: {len(common_list)}")
    print(f"EC1 classes: {len(hierarchical['ec1']['classes'])}")
    print(f"EC2 classes: {len(hierarchical['ec2']['classes'])}")
    print(f"EC3 classes: {len(hierarchical['ec3']['classes'])}")
    print(f"EC4 classes: {len(ec4_classes)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
