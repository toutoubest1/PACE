"""Missingness-pattern sensitivity of robust distances / weights (AE point 3)."""
import sys, warnings; warnings.filterwarnings('ignore')
import os; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT,'results','raw','studies'), exist_ok=True)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import pace.core as Mth
from sim.generate import ar1

rows = []; pair_rows = []
for rep in range(20):
    rng = np.random.default_rng(1000 + rep)
    n, p = 400, 300
    L = np.linalg.cholesky(ar1(p, 0.5))
    X = rng.standard_normal((n, p)) @ L.T
    mr = rng.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], n)
    M = rng.random((n, p)) >= mr[:, None]
    Xobs = np.where(M, X, 0.0)
    latent = np.sum(X ** 2, 1) / p
    # old weight
    rn = np.linalg.norm(Xobs, axis=1); med = np.median(rn); mad = np.median(np.abs(rn - med))
    c = max(1.345 * mad / 0.6745, 1.0); w_old = np.minimum(1, c / rn)
    mu, sd = Mth.robust_loc_scale(Xobs, M); Z = np.where(M, (Xobs - mu) / sd, 0.0)
    pi = M.mean(0)
    d2, meff = Mth.hajek_distance(Z, M, pi)
    d2zf = np.sum(Z ** 2, 1) / p
    d2ht = np.sum(np.where(M, Z, 0) ** 2 / pi, 1) / p
    w_new, _ = Mth.leverage_weight(d2, meff, p)
    w_zf, _ = Mth.leverage_weight(d2zf, np.full(n, float(p)), p)
    for i in range(n):
        rows.append(dict(rep=rep, miss=mr[i], latent=latent[i], zf_norm2=d2zf[i], ht=d2ht[i], hajek=d2[i],
                         w_old=w_old[i], w_zf=w_zf[i], w_new=w_new[i]))
    # matched pairs: similar latent magnitude (within 2%), missing rates 0.0 vs >=0.5
    lo = np.flatnonzero(mr == 0.0); hi = np.flatnonzero(mr >= 0.5)
    for i in lo:
        j = hi[np.argmin(np.abs(latent[hi] - latent[i]))]
        if abs(latent[j] - latent[i]) / latent[i] < 0.02:
            pair_rows.append(dict(rep=rep, lat_ratio=latent[j] / latent[i],
                                  dist_ratio_zf=d2zf[j] / d2zf[i], dist_ratio_hajek=d2[j] / d2[i],
                                  wratio_old=w_old[j] / w_old[i], wratio_zf=w_zf[j] / w_zf[i],
                                  wratio_new=w_new[j] / w_new[i]))
df = pd.DataFrame(rows); pr = pd.DataFrame(pair_rows)
tab = df.groupby('miss')[['latent', 'zf_norm2', 'ht', 'hajek', 'w_old', 'w_zf', 'w_new']].mean()
tab['frac_wzf<1'] = df.groupby('miss').apply(lambda g: (g.w_zf < 0.999).mean())
tab['frac_wnew<1'] = df.groupby('miss').apply(lambda g: (g.w_new < 0.999).mean())
print(tab.round(3).to_string())
cor = {k: spearmanr(df['miss'], df[k])[0] for k in ['zf_norm2', 'hajek', 'w_old', 'w_zf', 'w_new']}
corl = {k: spearmanr(df['latent'], df[k])[0] for k in ['zf_norm2', 'hajek', 'w_old', 'w_new']}
print('Spearman with missing rate:', {k: round(v, 3) for k, v in cor.items()})
print('Spearman with latent magnitude:', {k: round(v, 3) for k, v in corl.items()})
print('matched pairs (miss>=0.5 vs 0), n=%d' % len(pr)); print(pr.drop(columns='rep').describe().loc[['mean', '50%', 'std']].round(3).to_string())
tab.to_csv('' + os.path.join(ROOT, 'results', 'raw', 'studies') + '/pattern_weights.csv'); pr.to_csv('' + os.path.join(ROOT, 'results', 'raw', 'studies') + '/pattern_pairs.csv')
df.to_csv(os.path.join(ROOT,'results','raw','studies','pattern_rows_raw.csv'), index=False)
