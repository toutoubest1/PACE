#!/bin/sh
# Complete real-data workflow, from download to tables and figures.
# About 1 hour wall-clock with 2 parallel jobs (about 2 CPU-hours).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
JOBS=${JOBS:-2}
mkdir -p logs
sh scripts/00_download.sh                 # downloads + SHA-256 verification
python3 scripts/01_profile_candidates.py > logs/profile_candidates.log
python3 scripts/02_preprocess.py
for d in Daporinad Gemcitabine THZ-2-102-1 Bleomycin Trametinib SNX-2112; do
  for r in $(seq 0 19); do echo "$d $r"; done
done | xargs -P "$JOBS" -n 2 python3 scripts/03_run_split.py
python3 scripts/04_diagnostics.py > logs/diagnostics.log
python3 scripts/05_tables_figures.py > logs/tables.log
echo "real-data workflow complete"
