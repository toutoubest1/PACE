"""Part 2: outlier masking by missingness (zero-filled vs Hajek distance) + downstream estimation."""
import sys, warnings; warnings.filterwarnings('ignore')
import os; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT,'results','raw','studies'), exist_ok=True)
import numpy as np, pandas as pd
import pace.core as Mth
from sim.generate import ar1, metrics

det = []
for rep in range(20):
    rng = np.random.default_rng(2000 + rep)
    n, p = 400, 300
    L = np.linalg.cholesky(ar1(p, 0.5)); X = rng.standard_normal((n, p)) @ L.T
    out = rng.choice(n, 40, replace=False); isout = np.zeros(n, bool); isout[out] = True
    J = rng.choice(p, p // 2, replace=False)
    X[np.ix_(out, J)] += 1.5         # moderate coordinated leverage shift (latent d^2 ~ 2.1)
    mr = rng.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], n)
    M = rng.random((n, p)) >= mr[:, None]; Xobs = np.where(M, X, 0.0)
    rn = np.linalg.norm(Xobs, axis=1); med = np.median(rn); mad = np.median(np.abs(rn - med))
    w_old = np.minimum(1, max(1.345 * mad / 0.6745, 1.0) / rn)
    mu, sd = Mth.robust_loc_scale(Xobs, M); Z = np.where(M, (Xobs - mu) / sd, 0.0)
    d2, meff = Mth.hajek_distance(Z, M, M.mean(0))
    w_new, _ = Mth.leverage_weight(d2, meff, p)
    w_zf, _ = Mth.leverage_weight(np.sum(Z ** 2, 1) / p, np.full(n, float(p)), p)
    for i in range(n):
        det.append(dict(miss=mr[i], out=isout[i], w_old=w_old[i], w_zf=w_zf[i], w_new=w_new[i]))
d = pd.DataFrame(det)
g = d.groupby(['out', 'miss'])[['w_old', 'w_zf', 'w_new']].mean().round(3)
print('mean weight by (outlier?, row missing rate):'); print(g.to_string())
g.to_csv('' + os.path.join(ROOT, 'results', 'raw', 'studies') + '/pattern_detection.csv')
d.to_csv(os.path.join(ROOT,'results','raw','studies','pattern_detection_rows_raw.csv'), index=False)
