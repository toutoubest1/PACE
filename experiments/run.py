"""Generic, resumable simulation runner.
Example:
  python experiments/run.py ，out results/raw/final_main ，regimes none,yonly ，p 300 \
         block final ，reps 20 ，methods PACE,OLD-RLPD-MCP
One CSV per (regime, p, replicate) is written (checkpointing); results/merge happens in analysis/."""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from sim.generate import gen, metrics
from sim.registry import METHODS
from experiments.config import REGIMES, SEEDS, N

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--regimes", required=True)
ap.add_argument("--p", type=int, default=300)
ap.add_argument("--block", required=True, choices=list(SEEDS))
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--rep_start", type=int, default=0)
ap.add_argument("--methods", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
methods = a.methods.split(",")
for reg in a.regimes.split(","):
    for r in range(a.rep_start, a.reps):
        f = os.path.join(a.out, f"{reg}_p{a.p}_r{r:02d}.csv")
        if os.path.exists(f):
            done = set(pd.read_csv(f).Method)
            todo = [m for m in methods if m not in done]
            if not todo:
                continue
            rows = pd.read_csv(f).to_dict("records")
        else:
            todo, rows = methods, []
        seed = SEEDS[a.block] + r
        D = gen(n=N, p=a.p, seed=seed, **REGIMES[reg])
        for m in todo:
            t = time.time()
            b = METHODS[m](D)
            mt = metrics(b, D["beta"], D["Sigma"])
            mt.update(Method=m, regime=reg, p=a.p, rep=r, seed=seed, Time=time.time() - t)
            rows.append(mt)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(reg, a.p, r, "done", flush=True)
