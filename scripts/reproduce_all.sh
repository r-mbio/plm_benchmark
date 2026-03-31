#!/usr/bin/env bash
# ============================================================
# reproduce_all.sh
# ============================================================
# One-command full reproduction of all results and figures in:
# "Protein Language Models Outperform BLAST for Evolutionarily
#  Distant Enzymes: A Systematic Benchmark of EC Number
#  Prediction"
#
# What this script does:
#   1. Validates dependencies (Python, BLAST+, MMseqs2, GPU)
#   2. Runs cluster-based sequence splitting (MMseqs2)
#   3. Computes PLM embeddings (ESM2-650M, ESM2-3B, ProtT5)
#   4. Trains 1,296 benchmark conditions (3 replicates each)
#   5. Runs BLASTp baselines at all 4 thresholds
#   6. Runs distant eukaryote validation
#   7. Generates all main and supplementary figures
#   8. Generates all tables
#
# Expected runtime: 8–24 hours on a single NVIDIA A100 80GB GPU
# Disk usage: ~50 GB (embeddings: ~40 GB, models: ~8 GB)
#
# Usage:
#   bash scripts/reproduce_all.sh [--device cuda|cpu]
#                                 [--threads N]
#                                 [--skip-embeddings]
#                                 [--skip-training]
#                                 [--skip-blast]
#                                 [--figures-only]
# ============================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results}"
FIGURE_DIR="${FIGURE_DIR:-${PROJECT_ROOT}/figures}"
PYTHON="${PYTHON:-/home/r/rsathyamo/miniforge3/bin/python3}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-16}"
SKIP_EMBEDDINGS=0
SKIP_TRAINING=0
SKIP_BLAST=0
FIGURES_ONLY=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --device)          DEVICE="$2";        shift 2 ;;
    --threads)         THREADS="$2";       shift 2 ;;
    --skip-embeddings) SKIP_EMBEDDINGS=1;  shift   ;;
    --skip-training)   SKIP_TRAINING=1;    shift   ;;
    --skip-blast)      SKIP_BLAST=1;       shift   ;;
    --figures-only)    FIGURES_ONLY=1;     shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

export PROJECT_ROOT DATA_DIR RESULT_DIR PYTHON DEVICE THREADS

LOGFILE="${PROJECT_ROOT}/reproduce_all.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "============================================================"
echo " EC Benchmark Full Reproduction"
echo " Date: $(date)"
echo " Project: ${PROJECT_ROOT}"
echo " Python: ${PYTHON}"
echo " Device: ${DEVICE}"
echo " Log: ${LOGFILE}"
echo "============================================================"

# ── Step 0: Validate dependencies ────────────────────────────
echo ""
echo ">>> Step 0: Validating dependencies"

check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "  ERROR: '$1' not found on PATH. Please install it."
    exit 1
  fi
  echo "  OK: $1 ($(command -v "$1"))"
}

check_cmd "${PYTHON}"
check_cmd makeblastdb
check_cmd blastp
check_cmd mmseqs
check_cmd bc

# Check Python packages
"${PYTHON}" -c "
import importlib, sys
required = ['torch','numpy','pandas','sklearn','h5py','scipy','matplotlib','tqdm']
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(f'  ERROR: Missing Python packages: {missing}')
    sys.exit(1)
print('  OK: All required Python packages found.')
"

# Check GPU
if [[ "${DEVICE}" == "cuda" ]]; then
  "${PYTHON}" -c "
import torch
if not torch.cuda.is_available():
    print('  WARNING: CUDA requested but not available. Falling back to CPU.')
else:
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        print(f'  GPU {i}: {name} ({mem} GB)')
"
fi

if [[ "${FIGURES_ONLY}" -eq 1 ]]; then
  echo ""
  echo ">>> --figures-only: Skipping training, BLAST, and validation."
  goto_figures=1
else
  goto_figures=0
fi

# ── Step 1: Benchmark training ────────────────────────────────
if [[ "${goto_figures}" -eq 0 && "${SKIP_TRAINING}" -eq 0 ]]; then
  echo ""
  echo ">>> Step 1: Running benchmark training"
  SKIP_EMB_FLAG=""
  [[ "${SKIP_EMBEDDINGS}" -eq 1 ]] && SKIP_EMB_FLAG="--skip-embeddings"

  bash "${PROJECT_ROOT}/scripts/run_benchmark.sh" \
    --device "${DEVICE}" \
    --threads "${THREADS}" \
    ${SKIP_EMB_FLAG}
else
  echo ""
  echo ">>> Step 1: Skipping benchmark training (--skip-training or --figures-only)"
fi

# ── Step 2: BLAST baselines ───────────────────────────────────
if [[ "${goto_figures}" -eq 0 && "${SKIP_BLAST}" -eq 0 ]]; then
  echo ""
  echo ">>> Step 2: Running BLAST baselines"
  bash "${PROJECT_ROOT}/scripts/run_blast.sh" \
    --threshold all \
    --threads "${THREADS}"
else
  echo ""
  echo ">>> Step 2: Skipping BLAST (--skip-blast or --figures-only)"
fi

# ── Step 3: Distant eukaryote validation ─────────────────────
# NOTE: Distant eukaryote validation results are pre-computed and included in
# results/distant_eukaryote_validation.csv (13 organisms). Full re-validation
# requires downloading each organism's annotated proteome from UniProt and
# running PLM inference + BLASTp, which is outside the scope of this script.
echo ""
echo ">>> Step 3: Distant eukaryote validation (pre-computed — skipping re-run)"
echo "    Results: ${PROJECT_ROOT}/results/distant_eukaryote_validation.csv"

# ── Step 4: Generate figures ──────────────────────────────────
echo ""
echo ">>> Step 4: Generating figures"
mkdir -p "${FIGURE_DIR}"

# Use results if freshly computed, otherwise fall back to packaged results
BENCH_CSV="${RESULT_DIR}/benchmark_results_latest.csv"
BLAST_CSV="${RESULT_DIR}/blast_results_corrected.csv"
DISTANT_CSV="${RESULT_DIR}/distant_eukaryote_validation.csv"
ORGANISM_CSV="${RESULT_DIR}/organism_validation_plm_blast.csv"

# Fall back to packaged results if computed ones don't exist
[[ ! -f "${BENCH_CSV}" ]]    && BENCH_CSV="${PROJECT_ROOT}/results/benchmark_results_latest.csv"
[[ ! -f "${BLAST_CSV}" ]]    && BLAST_CSV="${PROJECT_ROOT}/results/blast_results_corrected.csv"
[[ ! -f "${DISTANT_CSV}" ]]  && DISTANT_CSV="${PROJECT_ROOT}/results/distant_eukaryote_validation.csv"
[[ ! -f "${ORGANISM_CSV}" ]] && ORGANISM_CSV="${PROJECT_ROOT}/results/organism_validation_plm_blast.csv"

for f in "${BENCH_CSV}" "${BLAST_CSV}" "${DISTANT_CSV}" "${ORGANISM_CSV}"; do
  if [[ ! -f "${f}" ]]; then
    echo "  ERROR: Required results file not found: ${f}"
    exit 1
  fi
done

"${PYTHON}" "${PROJECT_ROOT}/scripts/generate_figures.py" \
  --benchmark-csv "${BENCH_CSV}" \
  --blast-csv     "${BLAST_CSV}" \
  --distant-csv   "${DISTANT_CSV}" \
  --organism-csv  "${ORGANISM_CSV}" \
  --outdir        "${FIGURE_DIR}"

echo ""
echo "============================================================"
echo " Reproduction complete!"
echo " Figures: ${FIGURE_DIR}/main/ and ${FIGURE_DIR}/supplementary/"
echo " Results: ${RESULT_DIR}/"
echo " Log:     ${LOGFILE}"
echo " Date: $(date)"
echo "============================================================"
