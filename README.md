# EC Number Prediction Benchmark

**Protein Language Models Outperform BLAST for Evolutionarily Distant Enzymes: A Systematic Benchmark of EC Number Prediction**

Rajesh Sathyamoorthy

---

## Overview

This repository contains all code, scripts, and pre-computed results for a comprehensive benchmark of protein language models (PLMs) applied to Enzyme Commission (EC) number prediction. The benchmark evaluates:

- **3 PLMs**: ESM2-650M, ESM2-3B, ProtT5-XL
- **9 downstream architectures**: MLP, Deep MLP, Wide MLP, Attention MLP, CNN, ResNet, Multi-Head Attention MLP, Hybrid CNN-Transformer, Transformer Encoder
- **4 EC hierarchy levels**: EC1, EC2, EC3, EC4
- **4 sequence identity thresholds**: 30%, 50%, 70%, 90%
- **3 random seeds** per condition → **1,296 experimental conditions**, **3,888 trained models**
- **2 BLAST baselines**: train-only 90K and full SwissProt 520K
- **13 evolutionarily distant eukaryotes** for generalization testing

**Key findings:**
- Simple MLP classifiers achieve 96.6–98.0% accuracy at 50% identity, matching BLAST (±0.7 pp) on in-distribution data
- PLMs outperform BLAST by up to **+31.8 pp** for distant eukaryotes (Giardia lamblia)
- ESM2-650M is sufficient; ESM2-3B adds only 0.35 pp at 5× compute cost
- Transformer re-encoders fail at shared learning rate 1e-3 due to hyperparameter sensitivity


## Quick Start

### 1. Install Dependencies

```bash
# Conda environment (recommended)
conda create -n ec_benchmark python=3.10
conda activate ec_benchmark
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install esm transformers pandas numpy scikit-learn h5py scipy matplotlib seaborn tqdm

# BLAST+ (for BLAST baselines)
conda install -c bioconda blast mmseqs2
```

### 2. Reproduce Figures Only (pre-computed results)

If you just want to regenerate the publication figures from the pre-computed benchmark results:

```bash
python scripts/generate_figures.py \
  --benchmark-csv results/benchmark_results_latest.csv \
  --blast-csv     results/blast_results_corrected.csv \
  --distant-csv   results/distant_eukaryote_validation.csv \
  --outdir        figures/
```

Outputs saved to `figures/main/` and `figures/supplementary/` as PNG, SVG, and PDF.

### 3. Full Reproduction (requires GPU and data files)

Download the UniProt/SwissProt FASTA (2023 release) and filter for complete EC annotations:

```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
# Build EC label arrays (see data_notes/dataset_description.md)
python code/scripts/build_labels.py --fasta uniprot_sprot.fasta --output data/labels/
```

Then run the full reproduction:

```bash
bash scripts/reproduce_all.sh --device cuda --threads 32
```

This runs:
1. MMseqs2 clustering at 30/50/70/90% identity
2. PLM embedding computation (ESM2-650M, ESM2-3B, ProtT5)
3. 1,296 benchmark training runs (3 replicates each)
4. BLASTp baselines at all thresholds
5. All figure generation

**Estimated time:** 8–24 hours on a single NVIDIA A100 80GB GPU.

### 4. Run a Single Condition

```bash
python code/src/train.py \
  --plm esm2_650m \
  --architecture mlp \
  --ec-level ec4 \
  --threshold 50 \
  --seed 42 \
  --device cuda
```

---

## Data

### Input Data
- **Protein sequences and EC annotations**: UniProt/SwissProt (2023 release), filtered to 90,577 proteins with complete EC numbers
- **Clustering**: MMseqs2 at 30/50/70/90% sequence identity
- **Pre-computed embeddings**: Available at Zenodo (link to be added upon publication)

### Results Files (pre-computed, included in this repository)

| File | Description |
|---|---|
| `results/benchmark_results_latest.csv` | All 1,296 benchmark conditions (3 replicates), full metrics |
| `results/blast_results_corrected.csv` | BLASTp results at all 4 thresholds (90K train-matched database) |
| `results/distant_eukaryote_validation.csv` | 13-organism distant eukaryote validation (PLM vs BLAST) |
| `results/organism_validation_plm_blast.csv` | 9 held-out prokaryote cross-organism validation |

### Important Note on E. coli K-12

E. coli K-12 (taxon 83333) is included in the prokaryote validation but is **not a fully held-out organism**: approximately 20% of its proteome (~1,303/6,472 proteins) is present in the 90K training set. Results for E. coli should be interpreted with caution. The mean PLM–BLAST advantage excluding E. coli is **+17.2 pp** (vs +16.9 pp including it). See `data_notes/dataset_description.md` for details.

---

## Models

Pre-trained model weights are available at Zenodo (link to be added upon publication). Models are saved as PyTorch `.pt` checkpoints with corresponding `.meta.json` metadata files.

---

## Reproducibility Notes

- All training runs use deterministic seeding (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`)
- Three random seeds: 42, 123, 456
- BF16 mixed precision (no GradScaler needed; only FP16 requires scaling)
- Early stopping with patience=10, monitoring validation accuracy
- AdamW optimizer: lr=1e-3, weight_decay=0.01; CosineAnnealingLR schedule

---

## License

MIT License. See `LICENSE` for details.

