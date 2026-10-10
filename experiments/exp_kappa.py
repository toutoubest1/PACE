"""Population-level check of Prop. B: pattern-dependent weights induce a multiplicative bias
beta_w = kappa * beta* under row-heterogeneous MCAR (large n, unpenalised moment solution)."""
import sys, warnings; warnings.filterwarnings('ignore'); import os; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT,'results','raw','studies'), exist_ok=True)
import numpy as np, pandas as pd
import pace.core as Mth
from sim.generate import ar1
out = []
for rep in range(5):
    rng = np.random.default_rng(3000 + rep)
    n, p = 20000, 60
    L = np.linalg.cholesky(ar1(p, 0.5)); X = rng.standard_normal((n, p)) @ L.T
    beta = np.zeros(p); beta[:10] = rng.choice([-1, 1], 10) * rng.uniform(1, 2, 10)
    y = X @ beta + rng.standard_normal(n)
    mr = np.where(rng.random(n) < 0.5, 0.05, 0.5)           # row-heterogeneous MCAR
    M = rng.random((n, p)) >= mr[:, None]; Xo = np.where(M, X, 0.0); Mf = M.astype(float)
    # (a) OLD: zero-filled norm weight, column-pi IPW (exactly the old formulas)
    rn = np.linalg.norm(Xo, axis=1); med = np.median(rn); mad = np.median(np.abs(rn - med))
    w = np.minimum(1, max(1.345 * mad / 0.6745, 1) / rn)
    S = ((Xo * w[:, None]).T @ Xo / n) / (Mf.T @ Mf / n); r = (Xo.T @ (w * y) / n) / Mf.mean(0)
    b_old = np.linalg.solve(S, r)
    # (b) same old weight but pairwise-Hajek moments
    S2, r2, _ = Mth.hajek_moments(Xo, y, M, w); b_oldH = np.linalg.solve(S2, r2)
    # (c) pattern-free weight: same functional form applied to the Hajek distance
    mu, sd = Mth.robust_loc_scale(Xo, M); Z = np.where(M, (Xo - mu) / sd, 0.0)
    d2, meff = Mth.hajek_distance(Z, M, M.mean(0)); dd = np.sqrt(d2 * p)   # rescaled to norm units
    med = np.median(dd); mad = np.median(np.abs(dd - med))
    wn = np.minimum(1, max(1.345 * mad / 0.6745, 1) / dd)
    S3, r3, _ = Mth.hajek_moments(Xo, y, M, wn); b_new = np.linalg.solve(S3, r3)
    # (d) unweighted Hajek
    S4, r4, _ = Mth.hajek_moments(Xo, y, M, np.ones(n)); b_un = np.linalg.solve(S4, r4)
    S_ = beta != 0
    for nm, b in [('old weight + column IPW (old RLPD)', b_old), ('old weight + pairwise Hajek', b_oldH),
                  ('Hajek-distance weight + pairwise Hajek', b_new), ('unweighted pairwise Hajek', b_un)]:
        kappa = (b[S_] @ beta[S_]) / (beta[S_] @ beta[S_])
        out.append(dict(rep=rep, method=nm, kappa=kappa, rel_err=np.linalg.norm(b - beta) / np.linalg.norm(beta)))
df = pd.DataFrame(out)
t = df.groupby('method', sort=False)[['kappa', 'rel_err']].agg(['mean', 'std']).round(4)
print(t.to_string()); t.to_csv('' + os.path.join(ROOT, 'results', 'raw', 'studies') + '/kappa_bias.csv')
df.to_csv(os.path.join(ROOT,'results','raw','studies','kappa_raw.csv'), index=False)
