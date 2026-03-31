#!/usr/bin/env bash
# ============================================================
# run_blast.sh
# ============================================================
# Run BLASTp EC number prediction against the training-set
# database (fair 90K baseline) for all four sequence identity
# thresholds.
#
# Prerequisites:
#   - BLAST+ (makeblastdb, blastp) installed and on PATH
#   - Python ≥3.9 with pandas, numpy
#   - Benchmark train/test splits already computed
#     (run_benchmark.sh Step 1 must have completed)
#
# Usage:
#   bash scripts/run_blast.sh [--threshold 30|50|70|90|all]
#                             [--threads N]
#
# Outputs:
#   results/blast/blast_results_{THRESHOLD}pct.tsv
#   results/blast/blast_summary.csv
# ============================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results}"
PYTHON="${PYTHON:-$(command -v python3)}"
THREADS="${THREADS:-16}"
THRESHOLD="${THRESHOLD:-all}"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --threads)   THREADS="$2";   shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

BLAST_DIR="${RESULT_DIR}/blast"
mkdir -p "${BLAST_DIR}"

FASTA="${DATA_DIR}/sequences/proteins.fasta"
LABELS="${DATA_DIR}/labels/ec_labels.npz"

run_blast_for_threshold() {
  local THRESH="$1"
  echo ""
  echo "===================================================="
  echo " BLASTp at ${THRESH}% sequence identity threshold"
  echo "===================================================="

  TRAIN_FASTA="${BLAST_DIR}/train_${THRESH}pct.fasta"
  TEST_FASTA="${BLAST_DIR}/test_${THRESH}pct.fasta"
  DB_PATH="${BLAST_DIR}/blastdb_${THRESH}pct"
  RESULTS_FILE="${BLAST_DIR}/blast_results_${THRESH}pct.tsv"

  # Step A: Create train/test FASTA files
  echo "  [A] Creating train/test FASTA files for ${THRESH}%..."
  "${PYTHON}" "${PROJECT_ROOT}/code/scripts/create_blast_fasta.py" \
    --threshold "${THRESH}" \
    --fasta "${FASTA}" \
    --labels "${LABELS}" \
    --clusters "${DATA_DIR}/clusters/clusters_${THRESH}pct.tsv" \
    --train-out "${TRAIN_FASTA}" \
    --test-out "${TEST_FASTA}" \
    --seed 42

  # Step B: Build BLAST database
  echo "  [B] Building BLAST database..."
  makeblastdb \
    -in "${TRAIN_FASTA}" \
    -dbtype prot \
    -out "${DB_PATH}" \
    -parse_seqids
  echo "     Database: ${DB_PATH}"

  # Step C: Run BLASTp
  echo "  [C] Running BLASTp (this may take 10–30 minutes)..."
  blastp \
    -query "${TEST_FASTA}" \
    -db "${DB_PATH}" \
    -out "${RESULTS_FILE}" \
    -outfmt "6 qseqid sseqid pident evalue bitscore" \
    -max_target_seqs 5 \
    -num_threads "${THREADS}" \
    -evalue 1e-5
  echo "     Results: ${RESULTS_FILE}"

  # Step D: Evaluate
  echo "  [D] Evaluating BLAST accuracy..."
  "${PYTHON}" "${PROJECT_ROOT}/code/scripts/evaluate_blast.py" \
    --blast-results "${RESULTS_FILE}" \
    --labels "${LABELS}" \
    --clusters "${DATA_DIR}/clusters/clusters_${THRESH}pct.tsv" \
    --threshold "${THRESH}" \
    --output "${BLAST_DIR}/blast_eval_${THRESH}pct.json"
}

if [[ "${THRESHOLD}" == "all" ]]; then
  for T in 30 50 70 90; do
    run_blast_for_threshold "${T}"
  done
else
  run_blast_for_threshold "${THRESHOLD}"
fi

# Consolidate results
echo ""
echo "===================================================="
echo " Consolidating BLAST results..."
"${PYTHON}" - << 'PYEOF'
import json, glob, csv, os
blast_dir = os.environ.get("BLAST_DIR", "results/blast")
rows = []
for fpath in sorted(glob.glob(f"{blast_dir}/blast_eval_*pct.json")):
    with open(fpath) as f:
        d = json.load(f)
    rows.append(d)
if rows:
    keys = rows[0].keys()
    out = f"{blast_dir}/blast_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved summary: {out}")
else:
    print("  No evaluation JSON files found.")
PYEOF

echo ""
echo "BLAST run complete. Results in: ${BLAST_DIR}"
