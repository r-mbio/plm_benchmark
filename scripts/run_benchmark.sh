#!/usr/bin/env bash
# ============================================================
# run_benchmark.sh
# ============================================================
# Run the full EC number prediction benchmark:
#   - Computes embeddings for all 3 PLMs (if not cached)
#   - Trains 9 architectures × 4 EC levels × 4 thresholds × 3 seeds
#   - Total: 1,296 training runs (3,888 models across 3 replicates)
#
# Prerequisites:
#   - CUDA-capable GPU (A100 recommended; min 24 GB VRAM)
#   - Python ≥3.9 with torch, esm, transformers, scikit-learn, h5py, pandas
#   - MMseqs2 installed and on PATH
#   - UniProt/SwissProt FASTA at DATA_DIR/sequences/uniprot_sprot.fasta
#
# Usage:
#   bash scripts/run_benchmark.sh [--mode train|eval] [--device cuda|cpu]
#                                 [--threads N] [--skip-embeddings]
#
# Environment variables (can override defaults below):
#   PROJECT_ROOT   root of this repository
#   DATA_DIR       directory with sequences/, embeddings/, labels/
#   RESULT_DIR     where to write results
#   PYTHON         Python interpreter to use
# ============================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results}"
PYTHON="${PYTHON:-$(command -v python3)}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-16}"
SKIP_EMBEDDINGS="${SKIP_EMBEDDINGS:-0}"
MODE="${MODE:-train}"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)           MODE="$2";            shift 2 ;;
    --device)         DEVICE="$2";          shift 2 ;;
    --threads)        THREADS="$2";         shift 2 ;;
    --skip-embeddings) SKIP_EMBEDDINGS=1;   shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SRC_DIR="${PROJECT_ROOT}/code/src"
SCRIPTS_DIR="${PROJECT_ROOT}/code/scripts"

echo "============================================================"
echo " EC Benchmark — Full Run"
echo " Project root : ${PROJECT_ROOT}"
echo " Data dir     : ${DATA_DIR}"
echo " Results dir  : ${RESULT_DIR}"
echo " Device       : ${DEVICE}"
echo " Mode         : ${MODE}"
echo "============================================================"

mkdir -p "${RESULT_DIR}" "${DATA_DIR}/embeddings" "${DATA_DIR}/clusters"

# ── Step 1: Cluster sequences with MMseqs2 ───────────────────
echo ""
echo ">>> Step 1: Sequence clustering (MMseqs2)"
FASTA="${DATA_DIR}/sequences/proteins.fasta"

if [[ ! -f "${FASTA}" ]]; then
  echo "ERROR: Protein FASTA not found at ${FASTA}"
  echo "  Download from UniProt/SwissProt and place at the path above."
  exit 1
fi

for THRESH in 30 50 70 90; do
  CLUST_FILE="${DATA_DIR}/clusters/clusters_${THRESH}pct.tsv"
  if [[ -f "${CLUST_FILE}" ]]; then
    echo "  [skip] Clusters at ${THRESH}% already exist."
    continue
  fi
  echo "  Clustering at ${THRESH}%..."
  TMP_DIR=$(mktemp -d)
  mmseqs easy-cluster \
    "${FASTA}" \
    "${DATA_DIR}/clusters/cluster_result_${THRESH}pct" \
    "${TMP_DIR}" \
    --min-seq-id "$(echo "scale=2; ${THRESH}/100" | bc)" \
    --cov-mode 0 --coverage 0.8 \
    --cluster-mode 0 --threads "${THREADS}" \
    --quiet
  # Rename mmseqs output to expected TSV filename
  mv "${DATA_DIR}/clusters/cluster_result_${THRESH}pct_cluster.tsv" "${CLUST_FILE}"
  rm -rf "${TMP_DIR}"
  echo "  Done: ${CLUST_FILE}"
done

# ── Step 2: Compute PLM embeddings ──────────────────────────
if [[ "${SKIP_EMBEDDINGS}" -eq 0 ]]; then
  echo ""
  echo ">>> Step 2: Compute PLM embeddings"
  for PLM in esm2_650m esm2_3b prott5; do
    EMB_FILE="${DATA_DIR}/embeddings/${PLM}.h5"
    if [[ -f "${EMB_FILE}" ]]; then
      echo "  [skip] ${PLM} embeddings already exist."
      continue
    fi
    echo "  Computing ${PLM} embeddings..."
    "${PYTHON}" "${SCRIPTS_DIR}/compute_embeddings.py" \
      --plm "${PLM}" \
      --fasta "${FASTA}" \
      --output "${EMB_FILE}" \
      --device "${DEVICE}" \
      --batch-size 64
    echo "  Saved: ${EMB_FILE}"
  done
else
  echo ""
  echo ">>> Step 2: Skipping embedding computation (--skip-embeddings set)"
fi

# ── Step 3: Prepare labels ───────────────────────────────────
echo ""
echo ">>> Step 3: Prepare EC labels"
LABELS_FILE="${DATA_DIR}/labels/ec_labels.npz"
if [[ -f "${LABELS_FILE}" ]]; then
  echo "  [skip] Labels already prepared."
else
  "${PYTHON}" "${SCRIPTS_DIR}/prepare_labels.py" \
    --fasta "${FASTA}" \
    --output "${LABELS_FILE}"
fi

# ── Step 4: Run benchmark training ───────────────────────────
echo ""
echo ">>> Step 4: Training benchmark (1,296 conditions, 3 replicates each)"
echo "    This will take several hours on a single GPU."
echo "    Results will be written to: ${RESULT_DIR}/benchmark_results.csv"
echo ""

ARCHITECTURES="mlp deep_mlp wide_mlp attention_mlp cnn resnet multihead_attn hybrid_cnn_transformer transformer"
PLMS="esm2_650m esm2_3b prott5"
EC_LEVELS="ec1 ec2 ec3 ec4"
THRESHOLDS="30 50 70 90"
SEEDS="42 123 456"

TOTAL=0
for _ in ${ARCHITECTURES}; do
  for __ in ${PLMS}; do
    for ___ in ${EC_LEVELS}; do
      for ____ in ${THRESHOLDS}; do
        for _____ in ${SEEDS}; do
          TOTAL=$((TOTAL + 1))
        done
      done
    done
  done
done
echo "  Total training runs: ${TOTAL}"

"${PYTHON}" "${SRC_DIR}/../scripts/run_benchmark_grid.py" \
  --data-dir "${DATA_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --device "${DEVICE}" \
  --architectures ${ARCHITECTURES} \
  --plms ${PLMS} \
  --ec-levels ${EC_LEVELS} \
  --thresholds ${THRESHOLDS} \
  --seeds ${SEEDS} \
  --batch-size 2048 \
  --epochs 50 \
  --patience 10 \
  --lr 0.001 \
  --mode "${MODE}"

echo ""
echo ">>> Benchmark complete. Results at: ${RESULT_DIR}/benchmark_results.csv"
