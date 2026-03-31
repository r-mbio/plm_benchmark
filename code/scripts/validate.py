"""
External Validation on TrEMBL Organisms

Validates trained models on enzymes from organisms not seen during training.
Fetches sequences from UniProt TrEMBL and generates embeddings on-the-fly.

Usage:
    python validate.py                        # Validate best models
    python validate.py --plm esm2_3b          # Validate specific PLM
    python validate.py --organisms "Bacillus subtilis,Aspergillus niger"
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch

# Ensure local src/ imports resolve when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EC_LEVELS, MODEL_DIR, RESULT_DIR, setup_dirs, Config
from models import get_model


def _load_validation_organisms():
    """
    Load organism validation targets from data/validation/organisms.json.
    Falls back to a small static set if unavailable.
    """
    fallback = {
        "ecoli_k12": {"name": "Escherichia coli K-12", "taxid": 83333, "kingdom": "bacteria"},
        "haloferax": {"name": "Haloferax volcanii", "taxid": 2246, "kingdom": "archaea"},
        "sulfolobus": {"name": "Sulfolobus acidocaldarius", "taxid": 2285, "kingdom": "archaea"},
    }

    try:
        cfg = Config()
        data_dir = Path(cfg.paths.get("data_dir", "data"))
        org_json = data_dir / "validation" / "organisms.json"
        if not org_json.exists():
            return fallback

        with open(org_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        out = {}
        for key, entry in raw.items():
            taxid = int(entry.get("taxon_id", 0))
            t = str(entry.get("type", "unknown"))
            if "arch" in t.lower():
                kingdom = "archaea"
            elif "euk" in t.lower():
                kingdom = "eukaryota"
            else:
                kingdom = "bacteria"
            out[key] = {
                "name": entry.get("name", key),
                "taxid": taxid,
                "kingdom": kingdom,
            }
        return out or fallback
    except Exception:
        return fallback


VALIDATION_ORGANISMS = _load_validation_organisms()


def setup_directories():
    """Backward-compatible wrapper."""
    setup_dirs()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def get_result_path(stem: str) -> Path:
    """Return output JSON path for validation runs."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{stem}.json"


def get_model_path(plm: str, ec_level: str, threshold: int, architecture: str = "mlp") -> Path:
    """
    Resolve model path with a few filename patterns used across project revisions.
    """
    patterns = [
        MODEL_DIR / f"{plm}_{ec_level}_{threshold}pct_{architecture}.pt",
        MODEL_DIR / plm / f"{ec_level}_{threshold}pct_{architecture}.pt",
        MODEL_DIR / f"ec4_classifier_{threshold}pct.pt",
    ]
    for p in patterns:
        if p.exists():
            return p
    return patterns[0]


class UniProtFetcher:
    """Fetches enzyme sequences from UniProt TrEMBL."""
    
    BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
    
    def __init__(self, max_retries=3, timeout=60):
        self.max_retries = max_retries
        self.timeout = timeout
    
    def fetch_enzymes(self, taxid, limit=200, reviewed=False):
        """
        Fetch enzymes for a specific organism.
        
        Args:
            taxid: NCBI taxonomy ID
            limit: Maximum number of sequences
            reviewed: If True, fetch SwissProt; if False, fetch TrEMBL
        
        Returns:
            List of dicts with 'accession', 'sequence', 'ec' keys
        """
        review_str = "true" if reviewed else "false"
        
        params = {
            "query": f"organism_id:{taxid} AND reviewed:{review_str} AND ec:*",
            "format": "json",
            "fields": "accession,sequence,ec",
            "size": min(limit, 500),
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                results = []
                for entry in response.json().get("results", []):
                    seq = entry.get("sequence", {}).get("value", "")
                    
                    # Extract EC numbers
                    ec_numbers = []
                    protein_desc = entry.get("proteinDescription", {})
                    
                    # From recommended name
                    rec_name = protein_desc.get("recommendedName", {})
                    for ec in rec_name.get("ecNumbers", []):
                        ec_numbers.append(ec.get("value", ""))
                    
                    # From submission names
                    for sub_name in protein_desc.get("submissionNames", []):
                        for ec in sub_name.get("ecNumbers", []):
                            ec_numbers.append(ec.get("value", ""))
                    
                    # Filter valid entries
                    if seq and ec_numbers and 50 <= len(seq) <= 1500:
                        results.append({
                            "accession": entry.get("primaryAccession", ""),
                            "sequence": seq,
                            "ec": ec_numbers[0],  # Primary EC
                            "ec_all": ec_numbers,
                        })
                
                return results[:limit]
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(3)
                else:
                    print(f"    Failed to fetch: {e}")
                    return []
        
        return []


class EmbeddingGenerator:
    """Generates embeddings using ESM2 model."""
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def _load_model(self):
        if self.model is not None:
            return
        
        from transformers import AutoModel, AutoTokenizer
        
        print(f"  Loading {self.model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"  Model loaded on {self.device}")
    
    def generate(self, sequences, batch_size=8):
        """
        Generate embeddings for sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for inference
        
        Returns:
            numpy array of embeddings
        """
        self._load_model()
        
        embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling over sequence length
                hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)


def load_trained_model(plm, ec_level, threshold, architecture='mlp'):
    """Load a trained model from disk."""
    model_path = get_model_path(plm, ec_level, threshold, architecture)
    
    if not model_path.exists():
        return None, None
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    config = checkpoint.get('config', {})
    emb_dim = int(checkpoint.get('emb_dim', checkpoint.get('input_dim', 1280)))
    num_classes = int(checkpoint.get('num_classes', len(checkpoint.get('class_names', {})) or 2))

    # Support both newer and older get_model signatures.
    try:
        dropout = float(config.get('dropout', 0.3)) if isinstance(config, dict) else 0.3
        model = get_model(architecture, emb_dim, num_classes, dropout, 'cpu')
    except TypeError:
        model = get_model(architecture, emb_dim, num_classes, config)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint


def validate_organism(organism_name, organism_info, models, emb_generator, device):
    """
    Validate models on a single organism.
    
    Args:
        organism_name: Name of organism
        organism_info: Dict with taxid, kingdom
        models: Dict of loaded models by EC level
        emb_generator: EmbeddingGenerator instance
        device: torch device
    
    Returns:
        Dict with validation results
    """
    fetcher = UniProtFetcher()
    
    print(f"\n  {organism_name} ({organism_info['kingdom']})...")
    
    # Fetch enzymes
    enzymes = fetcher.fetch_enzymes(organism_info['taxid'], limit=200)
    
    if len(enzymes) < 10:
        print(f"    Skipped: only {len(enzymes)} enzymes found")
        return None
    
    print(f"    Fetched {len(enzymes)} enzymes")
    
    # Generate embeddings
    sequences = [e['sequence'] for e in enzymes]
    embeddings = emb_generator.generate(sequences)
    
    # Validate each EC level
    results = {
        'organism': organism_name,
        'kingdom': organism_info['kingdom'],
        'num_enzymes': len(enzymes),
        'ec_results': {},
    }
    
    for ec_level, (model, checkpoint) in models.items():
        if model is None:
            continue
        
        class_names = checkpoint.get('class_names', {})
        
        # Map enzyme EC numbers to model classes
        valid_indices = []
        true_labels = []
        
        for i, enzyme in enumerate(enzymes):
            ec_str = enzyme['ec']
            
            # Extract EC at appropriate level
            ec_parts = ec_str.split('.')
            if ec_level == 'ec1' and len(ec_parts) >= 1:
                ec_query = ec_parts[0]
            elif ec_level == 'ec2' and len(ec_parts) >= 2:
                ec_query = '.'.join(ec_parts[:2])
            elif ec_level == 'ec3' and len(ec_parts) >= 3:
                ec_query = '.'.join(ec_parts[:3])
            elif ec_level == 'ec4' and len(ec_parts) >= 4:
                ec_query = '.'.join(ec_parts[:4])
            else:
                continue
            
            # Find matching class
            for class_idx, class_name in class_names.items():
                if class_name == ec_query:
                    valid_indices.append(i)
                    true_labels.append(int(class_idx))
                    break
        
        if len(valid_indices) < 5:
            results['ec_results'][ec_level] = {
                'accuracy': None,
                'num_valid': len(valid_indices),
                'reason': 'insufficient_overlap',
            }
            continue
        
        # Predict
        X = torch.FloatTensor(embeddings[valid_indices]).to(device)
        model = model.to(device)
        
        with torch.no_grad():
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            # Top-5 accuracy
            top5_preds = logits.topk(5, dim=1).indices.cpu().numpy()
        
        y_true = np.array(true_labels)
        
        # Metrics
        acc = (preds == y_true).mean()
        top5_acc = np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))])
        
        results['ec_results'][ec_level] = {
            'accuracy': float(acc),
            'top5_accuracy': float(top5_acc),
            'num_valid': len(valid_indices),
        }
        
        print(f"    {ec_level.upper()}: acc={acc:.1%}, top5={top5_acc:.1%} (n={len(valid_indices)})")
    
    return results


def run_validation(plm='esm2_650m', threshold=30, architecture='mlp', 
                   organisms=None, verbose=True):
    """
    Run external validation.
    
    Args:
        plm: PLM to validate
        threshold: Sequence identity threshold
        architecture: Model architecture
        organisms: List of organism names (default: all)
        verbose: Print progress
    
    Returns:
        Dict with all validation results
    """
    setup_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Select organisms
    if organisms is None:
        organisms = list(VALIDATION_ORGANISMS.keys())
    
    # Load models for all EC levels
    print(f"\nLoading models: {plm}, {threshold}%, {architecture}")
    models = {}
    for ec_level in EC_LEVELS:
        model, checkpoint = load_trained_model(plm, ec_level, threshold, architecture)
        if model is not None:
            models[ec_level] = (model, checkpoint)
            print(f"  {ec_level.upper()}: {checkpoint['num_classes']} classes")
        else:
            print(f"  {ec_level.upper()}: not found")
            models[ec_level] = (None, None)
    
    # Initialize embedding generator
    if plm == 'esm2_650m':
        model_name = "facebook/esm2_t33_650M_UR50D"
    elif plm == 'esm2_3b':
        model_name = "facebook/esm2_t36_3B_UR50D"
    else:
        model_name = "facebook/esm2_t33_650M_UR50D"
    
    emb_generator = EmbeddingGenerator(model_name)
    
    # Validate each organism
    results = {
        'timestamp': datetime.now().isoformat(),
        'plm': plm,
        'threshold': threshold,
        'architecture': architecture,
        'organisms': [],
    }
    
    print("\nValidating on external organisms:")
    
    for org_name in organisms:
        if org_name not in VALIDATION_ORGANISMS:
            print(f"  Unknown organism: {org_name}")
            continue
        
        org_info = VALIDATION_ORGANISMS[org_name]
        org_result = validate_organism(org_name, org_info, models, emb_generator, device)
        
        if org_result is not None:
            results['organisms'].append(org_result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for ec_level in EC_LEVELS:
        accs = []
        for org in results['organisms']:
            ec_res = org['ec_results'].get(ec_level, {})
            if ec_res.get('accuracy') is not None:
                accs.append(ec_res['accuracy'])
        
        if accs:
            print(f"{ec_level.upper()}: mean={np.mean(accs):.1%}, "
                  f"min={np.min(accs):.1%}, max={np.max(accs):.1%} (n={len(accs)})")
    
    # Save results
    result_path = get_result_path(f'validation_{plm}_{threshold}pct')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {result_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate EC models on external organisms')
    parser.add_argument('--plm', type=str, default='esm2_650m',
                        help='PLM to validate')
    parser.add_argument('--threshold', type=int, default=30,
                        help='Sequence identity threshold')
    parser.add_argument('--architecture', type=str, default='mlp',
                        help='Model architecture')
    parser.add_argument('--organisms', type=str,
                        help='Comma-separated list of organisms')
    
    args = parser.parse_args()
    
    organisms = None
    if args.organisms:
        organisms = [o.strip() for o in args.organisms.split(',')]
    
    run_validation(
        plm=args.plm,
        threshold=args.threshold,
        architecture=args.architecture,
        organisms=organisms,
    )


if __name__ == '__main__':
    main()
