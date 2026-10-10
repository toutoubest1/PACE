"""Competitor estimators (Python re-implementations; see README for provenance)."""
import numpy as np
from .core import robust_loc_scale, hajek_moments, lpd
from .solver import fit_path_cv, _cd_weighted_l1


def huber_w(t, k=1.345):
    a = np.abs(t)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def ipw_lpd_lasso(Xobs, y, M, penalty="lasso"):
    """Non-robust IPW (pairwise-Hajek) + LPD + Lasso with moment-CV (Park et al. 2025 style)."""
    n, p = Xobs.shape
    U = M.copy()
    mu = np.array([Xobs[U[:, j], j].mean() if U[:, j].any() else 0 for j in range(p)])
    Xc = Xobs - mu
    yc = y - y.mean()
    one = np.ones(n)
    mf = lambda idx: hajek_moments(Xc[idx], yc[idx], U[idx], one[idx])
    b, _ = fit_path_cv(mf, n, 5, penalty=penalty, lpd_fn=lpd)
    return b


def _impute(Xobs, M, how="mean"):
    X = Xobs.copy().astype(float)
    for j in range(X.shape[1]):
        o = M[:, j]
        if o.any():
            X[~o, j] = X[o, j].mean() if how == "mean" else np.median(X[o, j])
        else:
            X[~o, j] = 0.0
    return X


def naive_lasso(Xobs, y, M):
    from sklearn.linear_model import LassoCV
    X = _impute(Xobs, M, "mean")
    m, s = X.mean(0), X.std(0) + 1e-8
    Xs = (X - m) / s
    mod = LassoCV(cv=5, alphas=50, max_iter=20000).fit(Xs, y - y.mean())
    return mod.coef_ / s


def mice_lasso(Xobs, y, M):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import Ridge, LassoCV
    p = Xobs.shape[1]
    Xn = np.where(M, Xobs, np.nan)
    imp = IterativeImputer(estimator=Ridge(alpha=1.0), max_iter=5, random_state=0,
                           n_nearest_features=min(30, p - 1))
    X = imp.fit_transform(Xn)
    m, s = X.mean(0), X.std(0) + 1e-8
    Xs = (X - m) / s
    mod = LassoCV(cv=5, alphas=50, max_iter=20000).fit(Xs, y - y.mean())
    return mod.coef_ / s


def huber_lasso(Xobs, y, M, k=1.345, n_irls=6, folds=5, n_lams=25, seed=0):
    """Huber-loss Lasso (Fan, Li & Wang 2017 / Sun, Zhou & Fan 2020 style) on median-imputed
    data. IRLS solver; lambda by K-fold CV of the held-out Huber loss (robust CV)."""
    pass
    n, p = Xobs.shape
    X = _impute(Xobs, M, "median")
    mu = np.median(X, 0); sd = 1.4826 * np.median(np.abs(X - mu), 0) + 1e-8
    Xs = (X - mu) / sd
    sy = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-8

    def fit(idx, lam, b0, a0):
        Xi, yi = Xs[idx], y[idx]
        b, a = b0.copy(), a0
        for _ in range(n_irls):
            res = yi - a - Xi @ b
            s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-8
            w = huber_w(res / s, k)
            a = np.sum(w * (yi - Xi @ b)) / w.sum()
            yc = yi - a
            if p <= len(idx):
                S = (Xi * w[:, None]).T @ Xi / len(idx); r = (Xi * w[:, None]).T @ yc / len(idx)
                S = S + 1e-6 * np.eye(p)
                b_new = _cd_weighted_l1(S, r, np.full(p, lam), b.copy())
            else:  # p > n: data-space coordinate descent (same objective up to weight scaling)
                from sklearn.linear_model import Lasso
                mdl = Lasso(alpha=lam / w.mean(), fit_intercept=False, warm_start=True, max_iter=3000, tol=1e-5)
                mdl.coef_ = b.copy()
                mdl.fit(Xi, yc, sample_weight=w)
                b_new = mdl.coef_.copy()
            if np.max(np.abs(b_new - b)) < 1e-5:
                b = b_new; break
            b = b_new
        return b, a

    lmax = np.max(np.abs(Xs.T @ (y - np.median(y)))) / n
    lams = np.logspace(np.log10(lmax), np.log10(0.01 * lmax), n_lams)
    rng = np.random.default_rng(seed); fid = rng.permutation(n) % folds
    loss = np.zeros((folds, n_lams))
    for f in range(folds):
        tr = np.flatnonzero(fid != f); te = np.flatnonzero(fid == f)
        b, a = np.zeros(p), np.median(y[tr])
        for l, lam in enumerate(lams):
            b, a = fit(tr, lam, b, a)
            res = (y[te] - a - Xs[te] @ b) / sy
            ar = np.abs(res)
            loss[f, l] = np.mean(np.where(ar <= k, 0.5 * ar ** 2, k * ar - 0.5 * k ** 2))
    lbest = lams[int(np.argmin(loss.mean(0)))]
    b, a = np.zeros(p), np.median(y)
    for lam in lams:
        b, a = fit(np.arange(n), lam, b, a)
        if lam <= lbest:
            break
    return b / sd


def sparse_lts(Xobs, y, M, alpha=0.75, n_starts=12, n_lams=12, seed=0, X_imp=None):
    """Sparse LTS (Alfons, Croux & Gelper 2013), median-imputed data; lambda via BIC
    of the reweighted fit (robustHD default criterion)."""
    from sklearn.linear_model import Lasso
    rng = np.random.default_rng(seed)
    n, p = Xobs.shape
    X = _impute(Xobs, M, "median") if X_imp is None else X_imp
    mu, sd = np.median(X, 0), 1.4826 * np.median(np.abs(X - np.median(X, 0)), 0) + 1e-8
    Xs = (X - mu) / sd
    yc = y - np.median(y)
    h = int(alpha * n)
    # lambda grid on the scale of robust correlations
    lmax = np.max(np.abs(Xs.T @ (yc - np.median(yc)))) / n
    lams = np.logspace(np.log10(0.5 * lmax), np.log10(0.01 * lmax), n_lams)
    best_crit, best_b = np.inf, np.zeros(p)
    for lam in lams:
        cands = []
        for st in range(n_starts):
            idx = rng.choice(n, 3, replace=False)
            m = Lasso(alpha=lam, max_iter=5000).fit(Xs[idx], yc[idx])
            for _ in range(2):
                res = (yc - m.intercept_ - Xs @ m.coef_) ** 2
                idx = np.argsort(res)[:h]
                m = Lasso(alpha=lam, max_iter=5000, warm_start=False).fit(Xs[idx], yc[idx])
            res = (yc - m.intercept_ - Xs @ m.coef_) ** 2
            cands.append((np.sort(res)[:h].sum(), idx))
        cands.sort(key=lambda z: z[0])
        bestobj, bb, bi = np.inf, None, 0.0
        for _, idx in cands[:3]:
            for _ in range(20):
                m = Lasso(alpha=lam, max_iter=5000).fit(Xs[idx], yc[idx])
                res = (yc - m.intercept_ - Xs @ m.coef_) ** 2
                new = np.sort(np.argsort(res)[:h])
                if np.array_equal(new, np.sort(idx)):
                    break
                idx = new
            obj = np.sort(res)[:h].sum()
            if obj < bestobj:
                bestobj, bb, bi = obj, m.coef_.copy(), m.intercept_
        # reweighting step
        res = yc - bi - Xs @ bb
        s = 1.4826 * np.median(np.abs(res)) + 1e-8
        keep = np.abs(res / s) <= 2.807  # 0.995 quantile
        m = Lasso(alpha=lam, max_iter=5000).fit(Xs[keep], yc[keep])
        res2 = yc[keep] - m.intercept_ - Xs[keep] @ m.coef_
        df = np.sum(m.coef_ != 0)
        sig = 1.4826 * np.median(np.abs(res2)) + 1e-8
        crit = np.log(sig) + df * np.log(keep.sum()) / keep.sum()
        if crit < best_crit:
            best_crit, best_b = crit, m.coef_ / sd
    return best_b


def filter_mice_slts(Xobs, y, M, tau=3.0):
    """Cellwise pipeline competitor (Toka et al. 2021 / cellLTS-style, high-dim surrogate):
    flag |z|>tau cells as missing -> iterative imputation -> sparse LTS."""
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import Ridge
    p = Xobs.shape[1]
    mu, sd = robust_loc_scale(Xobs, M)
    Z = (Xobs - mu) / sd
    U = M & ~(np.abs(Z) > tau)
    Xn = np.where(U, Xobs, np.nan)
    imp = IterativeImputer(estimator=Ridge(alpha=1.0), max_iter=5, random_state=0,
                           n_nearest_features=min(20, p - 1))
    X = imp.fit_transform(Xn)
    return sparse_lts(Xobs, y, U, X_imp=X)
