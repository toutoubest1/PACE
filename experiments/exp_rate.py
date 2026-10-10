"""Numerical check of Theorem 1: max-norm error of pairwise-Hajek moments vs n, pi, contamination."""
import sys, warnings; warnings.filterwarnings('ignore'); import os; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import pace.core as Mth
from sim.generate import ar1
p = 200; Sig = ar1(p, 0.5); L = np.linalg.cholesky(Sig); rows = []
for n in [200, 800, 3200]:
    for pi in [0.9, 0.7, 0.5, 0.3]:
        for eps in [0.0, 0.05, 0.1]:
            errs = []
            for rep in range(5):
                rng = np.random.default_rng(rep)
                X = rng.standard_normal((n, p)) @ L.T
                m = int(eps * n)
                X[:m] = 3.0 * np.sign(rng.standard_normal((m, 1)))   # coordinated outliers just inside the flag threshold (worst case)
                M = rng.random((n, p)) < pi
                U = M & (np.abs(X) <= 3.5)                              # usable cells (flag rule)
                S, _, _ = Mth.hajek_moments(X, np.zeros(n), U, np.ones(n))
                errs.append(np.max(np.abs(S - Sig)))
            rows.append(dict(n=n, pi=pi, eps=eps, maxerr=np.mean(errs)))
df = pd.DataFrame(rows)
df['sqrt(log p/n)/pi'] = np.sqrt(np.log(p) / df.n) / df.pi
print(df.round(4).to_string()); df.to_csv('' + os.path.join(ROOT, 'results', 'raw', 'studies') + '/rate_check.csv', index=False)
for eps in [0.0]:
    d = df[df.eps == eps]; c = np.polyfit(np.log(d['sqrt(log p/n)/pi']), np.log(d.maxerr), 1)
    print('eps=0: log-log slope of maxerr on sqrt(log p/n)/pi =', round(c[0], 3))
