"""PSD-stabiliser ablation inside PACE: LPD everywhere vs eigenvalue clipping at the final fit (PACE)
vs LPD with intensity chosen by downstream moment-CV risk.  Final seeds 7000-7009, p=300."""
import os, sys, time, warnings; warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import pandas as pd
from sim.generate import gen, metrics
from pace import pace_fit
from experiments.config import REGIMES, SEEDS
out = os.path.join(ROOT, 'results', 'raw', 'studies'); os.makedirs(out, exist_ok=True)
rows = []
for reg in ["none", "yonly", "xy", "cell6"]:
    for r in range(10):
        D = gen(n=200, p=300, seed=SEEDS["final"] + r, **REGIMES[reg])
        for st in ["lpd", "clip_final", "cvalpha"]:
            t = time.time()
            b = pace_fit(D["Xobs"], D["y"], D["M"], stab=st)
            m = metrics(b, D["beta"], D["Sigma"]); m.update(regime=reg, stab=st, rep=r, Time=time.time() - t)
            rows.append(m)
    print(reg, flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(out, 'stabiliser_raw.csv'), index=False)
print(df.groupby(['regime', 'stab'])[['L2', 'F1', 'FDR', 'Size']].mean().round(3).to_string())
