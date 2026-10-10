"""Shared helpers for the real-data analysis (protocol Sections 2-4)."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The frozen PACE package: repository root (../pace) if present, otherwise the vendored copy.
_REPO = os.path.dirname(ROOT)
sys.path.insert(0, _REPO if os.path.isdir(os.path.join(_REPO, "pace")) else os.path.join(ROOT, "vendor"))
from pace import fit_pace, pace, old_rlpd, old_lpd        # noqa: E402
from pace import competitors as C                          # noqa: E402

PROC = os.path.join(ROOT, "data_proc")
PANEL = ["Daporinad", "Gemcitabine", "THZ-2-102-1", "Bleomycin", "Trametinib", "SNX-2112"]
R_SPLITS = 20
SEED0 = 8000

# label in paper -> callable(Xobs, y, M) returning slopes
METHODS = {
    "PACE-CF5": lambda X, y, M: fit_pace(X, y, M),
    "PACE-Full": lambda X, y, M: pace(X, y, M, mode="full"),
    "RLPD-MCP": lambda X, y, M: old_rlpd(X, y, M, "mcp")[0],
    "RLPD-Lasso": lambda X, y, M: old_rlpd(X, y, M, "lasso")[0],
    "LPD-Lasso": lambda X, y, M: old_lpd(X, y, M, "lasso")[0],
    "IPW-LPD-Lasso": lambda X, y, M: C.ipw_lpd_lasso(X, y, M),
    "Huber-Lasso": lambda X, y, M: C.huber_lasso(X, y, M),
    "Sparse LTS": lambda X, y, M: C.sparse_lts(X, y, M),
    "Filter-MICE-SparseLTS": lambda X, y, M: C.filter_mice_slts(X, y, M),
    "MICE-Lasso": lambda X, y, M: C.mice_lasso(X, y, M),
    "Naive-Lasso": lambda X, y, M: C.naive_lasso(X, y, M),
}
ORDER = list(METHODS) + ["Null"]


def load():
    X = pd.read_csv(os.path.join(PROC, "X500.csv"), index_col=0)
    A = pd.read_csv(os.path.join(PROC, "auc_panel.csv"), index_col=0)
    L = pd.read_csv(os.path.join(PROC, "lines.csv"), index_col=0)
    return X, A, L


def drug_data(X, A, drug):
    y = A[drug].dropna()
    return X.loc[y.index].values.astype(float), y.values.astype(float), list(y.index)


def split(n, r):
    perm = np.random.default_rng(SEED0 + r).permutation(n)
    k = int(np.floor(0.8 * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def standardiser(Xtr_nan):
    """Training-median centring and 1.4826*MAD scaling of observed values (SD if MAD = 0)."""
    p = Xtr_nan.shape[1]
    med = np.zeros(p); sc = np.ones(p)
    for j in range(p):
        v = Xtr_nan[~np.isnan(Xtr_nan[:, j]), j]
        if v.size == 0:
            continue
        med[j] = np.median(v)
        s = 1.4826 * np.median(np.abs(v - med[j]))
        if not s > 0:
            s = np.std(v) if np.std(v) > 0 else 1.0
        sc[j] = s
    return med, sc


def to_input(Xnan, med, sc):
    """Standardised predictors with zeros at missing cells, and the observation mask."""
    M = ~np.isnan(Xnan)
    Z = np.where(M, (np.nan_to_num(Xnan) - med) / sc, 0.0)
    return Z, M


def mice_test_imputer(Xtr_nan):
    """Secondary prediction rule: IterativeImputer with MICE-Lasso's frozen settings."""
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import Ridge
    p = Xtr_nan.shape[1]
    return IterativeImputer(estimator=Ridge(alpha=1.0), max_iter=5, random_state=0,
                            n_nearest_features=min(30, p - 1)).fit(Xtr_nan)


def predict(beta, Ztr, ytr_c, Zte):
    """Common prediction rule: robust intercept = median training residual."""
    a = np.median(ytr_c - Ztr @ beta)
    return a + Zte @ beta


def metrics(y, yhat):
    from scipy.stats import spearmanr
    e = y - yhat
    rho = spearmanr(yhat, y).statistic if np.std(yhat) > 0 else np.nan
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))), Spearman=float(rho))
