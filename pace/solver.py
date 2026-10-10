"""Penalised quadratic solvers for Lasso and MCP, with EBIC and moment cross-validation.

Solves:
    min_b 0.5 * b.T @ S @ b - r.T @ b + sum_j lam_j * abs(b_j)

The covariance matrix is standardised to unit diagonal before optimisation.
MCP is fitted using local linear approximation (LLA).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _sweep(S, r, lamv, b, g, idx):
    maxd = 0.0
    for jj in range(idx.shape[0]):
        j = idx[jj]
        bj = b[j]
        z = r[j] - g[j] + S[j, j] * bj
        if z > lamv[j]:
            nb = (z - lamv[j]) / S[j, j]
        elif z < -lamv[j]:
            nb = (z + lamv[j]) / S[j, j]
        else:
            nb = 0.0
        d = nb - bj
        if d != 0.0:
            for k in range(S.shape[0]):
                g[k] += S[k, j] * d
            b[j] = nb
            ad = abs(d) * np.sqrt(S[j, j])
            if ad > maxd:
                maxd = ad
    return maxd


@njit(cache=True)
def _cd_weighted_l1(S, r, lamv, b, max_iter=1000, tol=1e-6):
    """active-set cyclic coordinate descent"""
    p = S.shape[0]
    g = S @ b
    allidx = np.arange(p)
    for outer in range(100):
        maxd = _sweep(S, r, lamv, b, g, allidx)
        if maxd < tol:
            break
        act = np.flatnonzero(b != 0.0)
        for it in range(max_iter):
            if _sweep(S, r, lamv, b, g, act) < tol:
                break
    return b


def _standardize(S, r):
    sd = np.sqrt(np.maximum(np.diag(S), 1e-10))
    C = S / np.outer(sd, sd)
    C = 0.5 * (C + C.T)
    return C, r / sd, sd


def lasso_path(S, r, lams, penalty="lasso", gamma=3.0, n_lla=3, max_df=None):
    """Return list of coefficient vectors (original scale) along lams (decreasing).
    If max_df is given, the path is truncated once the support exceeds max_df
    (remaining entries repeat the last solution)."""
    C, rc, sd = _standardize(S, r)
    p = C.shape[0]
    b = np.zeros(p)
    out = []
    for lam in lams:
        if max_df is not None and len(out) and np.sum(out[-1] != 0) > max_df:
            out.append(out[-1]); continue
        lamv = np.full(p, lam)
        b = _cd_weighted_l1(C, rc, lamv, b.copy())
        if penalty == "mcp":
            bb = b.copy()
            for _ in range(n_lla):
                lamv2 = np.maximum(lam - np.abs(bb) / gamma, 0.0)
                bb_new = _cd_weighted_l1(C, rc, lamv2, bb.copy())
                if np.max(np.abs(bb_new - bb)) < 1e-6:
                    bb = bb_new
                    break
                bb = bb_new
            out.append(bb / sd)
        else:
            out.append(b / sd)
    return out


def lam_grid(r_corr_max, n_lams=40, max_ratio=1.0, min_ratio=1e-3):
    return np.logspace(np.log10(r_corr_max * max_ratio), np.log10(r_corr_max * min_ratio), n_lams)


def fit_path_select(S, r, s_yy, n, p_dim=None, penalty="lasso", n_lams=40, gamma_ebic=None,
                    min_ratio=1e-3, max_df=None):
    """Path + EBIC on the estimated weighted residual variance
        Q(b) = s_yy - 2 r'b + b'S b,
    criterion  n log Q + df (log n + 2 gamma log p),  gamma = Chen-Chen rule if None."""
    p = S.shape[0]
    if p_dim is None:
        p_dim = p
    if gamma_ebic is None:
        kappa = np.log(p_dim) / np.log(n)
        gamma_ebic = max(0.0, 1.0 - 1.0 / (2.0 * kappa))
    if max_df is None:
        max_df = int(0.5 * n)
    C, rc, sd = _standardize(S, r)
    lmax = np.max(np.abs(rc))
    lams = lam_grid(lmax, n_lams, 1.0, min_ratio)
    betas = lasso_path(S, r, lams, penalty=penalty, max_df=max_df)
    best, best_b, best_lam = np.inf, np.zeros(p), lams[0]
    floor = 1e-3 * s_yy
    for lam, b in zip(lams, betas):
        df = int(np.sum(np.abs(b) > 1e-8))
        if df > max_df:
            break
        Q = s_yy - 2 * r @ b + b @ S @ b
        Q = max(Q, floor)
        crit = n * np.log(Q) + df * (np.log(n) + 2 * gamma_ebic * np.log(p_dim))
        if crit < best:
            best, best_b, best_lam = crit, b, lam
    return best_b, best_lam


def old_path(S, r, lams, penalty="lasso", a=3.7, gamma=3.0, n_lla=10):
    """Regularisation path with penalties on the original coefficient scale.
    SCAD and MCP use local linear approximation."""
    C, rc, sd = _standardize(S, r)
    p = C.shape[0]
    b = np.zeros(p)
    out = []
    for lam in lams:
        b = _cd_weighted_l1(C, rc, np.full(p, lam) / sd, b.copy())
        if penalty == "lasso":
            out.append(b / sd)
            continue
        bb = b.copy()
        for _ in range(n_lla):
            ab = np.abs(bb / sd)
            if penalty == "scad":
                w = np.where(ab <= lam, lam, np.where(ab <= a * lam, (a * lam - ab) / (a - 1), 0.0))
            else:
                w = np.where(ab <= gamma * lam, np.maximum(lam - ab / gamma, 0.0), 0.0)
            nb = _cd_weighted_l1(C, rc, w / sd, bb.copy())
            if np.max(np.abs(nb - bb)) < 1e-6:
                bb = nb
                break
            bb = nb
        out.append(bb / sd)
    return out


def old_ebic_select(S, r, n, penalty="lasso", gamma_ebic=0.0, n_lams=40, min_ratio=1e-4, max_ratio=0.5):
    """Select the baseline penalty using a penalised quadratic loss and EBIC term."""
    p = S.shape[0]
    lam_max = np.max(np.abs(r))
    lams = np.logspace(np.log10(lam_max * max_ratio), np.log10(lam_max * min_ratio), n_lams)
    betas = old_path(S, r, lams, penalty=penalty)
    best, bb = np.inf, None
    for b in betas:
        s = np.sum(np.abs(b) > 1e-6)
        if s == 0 or s >= n:
            continue
        val = 0.5 * b @ S @ b - r @ b + (s * np.log(n) + 2 * gamma_ebic * s * np.log(p)) / n
        if val < best:
            best, bb = val, b
    if bb is None:
        bb = betas[-1]
    return bb


def fit_path_cv(moment_fn, n, folds, penalty="lasso", n_lams=40, min_ratio=1e-3, one_se=True,
                lpd_fn=None, seed=0):
    """Moment cross-validation for IPW quadratic surrogates.
    moment_fn(idx) -> (S, r, syy) computed from rows idx (raw, possibly indefinite).
    Training moments are PD-stabilised by lpd_fn; held-out loss is the unbiased raw-moment
    estimate of the weighted prediction risk  syy - 2 r'b + b'S b."""
    S, r, syy = moment_fn(np.arange(n))
    Sp, _ = lpd_fn(S)
    C, rc, sd = _standardize(Sp, r)
    lams = lam_grid(np.max(np.abs(rc)), n_lams, 1.0, min_ratio)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fid = np.empty(n, int); fid[perm] = np.arange(n) % folds
    L = np.zeros((folds, n_lams))
    for k in range(folds):
        tr = np.flatnonzero(fid != k); te = np.flatnonzero(fid == k)
        Str, rtr, _ = moment_fn(tr)
        Strp, _ = lpd_fn(Str)
        Ste, rte, syyte = moment_fn(te)
        betas = lasso_path(Strp, rtr, lams, penalty=penalty, max_df=max(n // 3, 10))
        for l, b in enumerate(betas):
            L[k, l] = syyte - 2 * rte @ b + b @ Ste @ b
    m = L.mean(0); se = L.std(0, ddof=1) / np.sqrt(folds)
    i = int(np.argmin(m))
    if one_se:
        ok = np.flatnonzero(m <= m[i] + se[i])
        i = int(ok.min())  # largest lambda within 1 SE
    betas = lasso_path(Sp, r, lams[: i + 1], penalty=penalty)
    fit_path_cv.last_loss = float(m[i])
    return betas[-1], lams[i]
