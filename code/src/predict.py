"""
EC Prediction Script
====================
Predict EC numbers for new protein sequences using trained models.

Usage:
    # Predict from FASTA file
    python predict.py --fasta proteins.fasta --output predictions.csv
    
    # Predict from embeddings
    python predict.py --embeddings embeddings.h5 --output predictions.csv

Author: Rajesh Sathyamoorthy
"""

import argparse
import json
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from models import get_model


def load_model(model_path: Path, ec_classes_path: Path, device: str = 'cuda'):
    """
    Load trained model and class mappings.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        ec_classes_path: Path to EC classes JSON file
        device: Device to load model on
    
    Returns:
        Tuple of (model, ec_classes, config)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load EC classes
    with open(ec_classes_path) as f:
        ec_classes = json.load(f)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 1280)
    num_classes = len(ec_classes)
    arch = config.get('architecture', 'mlp')
    
    # Create and load model
    model = get_model(arch, input_dim, num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, ec_classes, config


def predict_from_embeddings(
    model: torch.nn.Module,
    embeddings: Dict[str, np.ndarray],
    ec_classes: List[str],
    device: str = 'cuda',
    batch_size: int = 1024,
    top_k: int = 5
) -> Dict[str, Dict]:
    """
    Predict EC numbers from pre-computed embeddings.
    
    Args:
        model: Trained model
        embeddings: Dict mapping protein_id -> embedding
        ec_classes: List of EC class names
        device: Device for inference
        batch_size: Batch size for inference
        top_k: Number of top predictions to return
    
    Returns:
        Dict mapping protein_id -> prediction results
    """
    model.eval()
    
    protein_ids = list(embeddings.keys())
    all_embeddings = np.array([embeddings[pid] for pid in protein_ids], dtype=np.float32)
    
    predictions = {}
    
    with torch.no_grad():
        for i in range(0, len(protein_ids), batch_size):
            batch_ids = protein_ids[i:i+batch_size]
            batch_emb = torch.FloatTensor(all_embeddings[i:i+batch_size]).to(device)
            
            logits = model(batch_emb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            for j, pid in enumerate(batch_ids):
                top_indices = np.argsort(probs[j])[::-1][:top_k]
                
                predictions[pid] = {
                    'predicted_ec': ec_classes[top_indices[0]],
                    'confidence': float(probs[j, top_indices[0]]),
                    'top_predictions': [
                        {'ec': ec_classes[idx], 'probability': float(probs[j, idx])}
                        for idx in top_indices
                    ]
                }
    
    return predictions


def predict_binary(
    model: torch.nn.Module,
    embeddings: Dict[str, np.ndarray],
    device: str = 'cuda',
    batch_size: int = 1024,
    threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    Binary enzyme/non-enzyme prediction.
    
    Args:
        model: Trained binary classifier
        embeddings: Dict mapping protein_id -> embedding
        device: Device for inference
        batch_size: Batch size
        threshold: Classification threshold
    
    Returns:
        Dict mapping protein_id -> prediction results
    """
    model.eval()
    
    protein_ids = list(embeddings.keys())
    all_embeddings = np.array([embeddings[pid] for pid in protein_ids], dtype=np.float32)
    
    predictions = {}
    
    with torch.no_grad():
        for i in range(0, len(protein_ids), batch_size):
            batch_ids = protein_ids[i:i+batch_size]
            batch_emb = torch.FloatTensor(all_embeddings[i:i+batch_size]).to(device)
            
            logits = model(batch_emb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            for j, pid in enumerate(batch_ids):
                enzyme_prob = probs[j, 1]  # Assuming class 1 is enzyme
                
                predictions[pid] = {
                    'is_enzyme': enzyme_prob >= threshold,
                    'enzyme_probability': float(enzyme_prob),
                    'confidence': float(max(enzyme_prob, 1 - enzyme_prob))
                }
    
    return predictions


def load_embeddings_from_h5(filepath: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """Load embeddings from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        
        # Find ID key
        id_key = None
        for candidate in ['ids', 'uniprot_ids', 'protein_ids']:
            if candidate in keys:
                id_key = candidate
                break
        
        if id_key is None:
            raise KeyError(f"No ID key found. Available: {keys}")
        
        emb_key = 'embeddings' if 'embeddings' in keys else [k for k in keys if k != id_key][0]
        
        ids = f[id_key][:]
        embeddings = f[emb_key][:]
    
    # Decode IDs if needed
    if isinstance(ids[0], bytes):
        ids = [x.decode('utf-8') for x in ids]
    
    return dict(zip(ids, embeddings)), embeddings.shape[1]


def main():
    parser = argparse.ArgumentParser(description='EC Number Prediction')
    parser.add_argument('--embeddings', type=Path, required=True, help='HDF5 file with embeddings')
    parser.add_argument('--model', type=Path, default=Path('models/ec_classifier.pt'), help='Model checkpoint')
    parser.add_argument('--ec-classes', type=Path, default=Path('models/ec_classes.json'), help='EC classes JSON')
    parser.add_argument('--output', type=Path, required=True, help='Output CSV file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    model, ec_classes, config = load_model(args.model, args.ec_classes, device)
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    embeddings, dim = load_embeddings_from_h5(args.embeddings)
    print(f"Loaded {len(embeddings)} proteins with {dim}-dim embeddings")
    
    # Predict
    print("Running predictions...")
    predictions = predict_from_embeddings(model, embeddings, ec_classes, device, top_k=args.top_k)
    
    # Save results
    import pandas as pd
    rows = []
    for pid, pred in predictions.items():
        row = {
            'protein_id': pid,
            'predicted_ec': pred['predicted_ec'],
            'confidence': pred['confidence']
        }
        for i, top_pred in enumerate(pred['top_predictions']):
            row[f'ec_rank{i+1}'] = top_pred['ec']
            row[f'prob_rank{i+1}'] = top_pred['probability']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == '__main__':
    main()
