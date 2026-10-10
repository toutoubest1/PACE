#!/bin/sh
# Run simulations and the real-data workflow; outputs remain local.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
bash experiments/run_all_final.sh
python3 analysis/make_tables.py
python3 analysis/make_figures.py
sh real_data/scripts/run_all.sh
python3 paper_tools/make_tex_tables.py
python3 paper_tools/make_figures.py
