#!/bin/sh
# All final (held-out seed block "final" = 7000+r) runs .  Lines are independent and can be
# run in parallel on different cores.  run.py is resumable.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
. experiments/final_methods.txt
ALLREG=none,xonly,xlev,yonly,xy,t2err,cell6,cell4,xy_m10,xy_m30,xy_m40,xy_hetcol,xy_hetrow,xy_mar,xy_mnar
# 1. main comparison + ablations, p = 300 (20 reps)
python3 experiments/run.py --out results/raw/final_main --regimes $ALLREG --p 300 --block final --reps 20 --methods $MAIN,PACE-SC-CF5
# 2. dimension study p in {100,500,1000} (20/10/10 reps)
python3 experiments/run.py --out results/raw/final_pgrid --regimes yonly,none,xy,cell6 --p 100  --block final --reps 20 --methods $PGRID,PACE-SC-CF5
python3 experiments/run.py --out results/raw/final_pgrid --regimes yonly,none,xy,cell6 --p 500  --block final --reps 10 --methods $PGRID,PACE-SC-CF5
python3 experiments/run.py --out results/raw/final_pgrid --regimes yonly,none,xy,cell6 --p 1000 --block final --reps 10 --methods $PGRID,PACE-SC-CF5
# 3. auxiliary studies
python3 experiments/exp_pattern.py
python3 experiments/exp_pattern2.py
python3 experiments/exp_kappa.py
python3 experiments/exp_rate.py
python3 experiments/exp_stab.py
# 4. development runs of validation round 2 (design record; seeds 2000+r)
# python3 experiments/run.py --out results/raw/dev_pilot --regimes yonly,none,xy,cell6,xonly,t2err --p 300 --block dev_pilot --reps 10 --methods PACE,PACE-pilot-screen,PACE-pilot-screen+cstep,PACE-pilot-screen_cstep,PACE-pilot-screen_cstep-K38,PACE-pilot-screen_cstep+cstep
# python3 experiments/run.py --out results/raw/dev_crossfit --regimes yonly,none,xy,cell6,xonly,t2err --p 300 --block dev_pilot --reps 10 --methods PACE,PACE-split,PACE-split-ls,PACE-halfsample,PACE-crossfit,PACE-crossfit-ls,PACE-crossfit5,PACE-crossfit5-ls
