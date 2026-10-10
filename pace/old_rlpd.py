"""Radial-weighting baseline estimators (RLPD and LPD).

The default settings correct two defects in the earlier implementation.

Defects of the released code (Nonconvex-Penalization-main):
 (1) covariance_estimators.py computes pi_xx = (mask.T @ mask)/n with a *boolean* mask.  NumPy then
     returns a boolean matrix product, so pi_jk = 1/n (or the 1e-4 floor) instead of the pairwise
     observation rate; Sigma_hat is inflated ~n*pi_jk (~160x) while rho_hat is not, and the fitted
     coefficients are essentially zero.  Fixed here by casting the mask to float (buggy=True
     reproduces the defect for diagnostics only).
 (2) shrinkage.py uses an absolute eigenvalue floor 1e-4 although the manuscript states a relative
     floor eta*mu with eta = 0.02.  Default here: relative (eps_mode="relative").
"""
import numpy as np
from numpy.linalg import eigvalsh
from .solver import old_ebic_select


def old_rlpd(Xobs, y, M, penalty="lasso", eps_mode="relative", buggy=False, robust=True):
    """Radial-weighted baseline: zero-filled-norm weight min(1,c/||x_obs||), MAD-calibrated c,
    IPW moments, LPD boundary shrinkage, EBIC(gamma=0) on the quadratic loss.
    eps_mode="relative": eigenvalue floor 0.02*mu as stated in the manuscript (default here);
    eps_mode="absolute": 1e-4 as in the released code.
    buggy=True reproduces the released code exactly: the boolean mask makes  mask.T @ mask  a
    boolean product, so pi_jk becomes 1/n (or the 1e-4 floor) and Sigma is inflated ~n*pi_jk times."""
    n, p = Xobs.shape
    rn = np.linalg.norm(Xobs, axis=1)
    med = np.median(rn); mad = np.median(np.abs(rn - med))
    c = max(1.345 * mad / 0.6745 if mad > 1e-8 else 3.0, 1.0)
    w = np.minimum(1.0, c / (rn + 1e-8)) if robust else np.ones(n)
    Mf = M.astype(float)
    Xw = Xobs * np.sqrt(w[:, None])
    pxx = (M.T @ M) / n if buggy else (Mf.T @ Mf / n)
    pxx = np.maximum(pxx, 1e-4)
    S = (Xw.T @ Xw / n) / pxx
    S = 0.5 * (S + S.T)
    r = (Xobs.T @ (w * y) / n) / np.maximum(Mf.mean(0), 1e-4)
    mu = np.trace(S) / p; lmin = eigvalsh(S)[0]
    eps = 0.02 * mu if eps_mode == "relative" else 1e-4
    a = 1.0 if lmin >= eps else float(np.clip((mu - eps) / (mu - lmin), 0, 1))
    S = a * S + (1 - a) * mu * np.eye(p)
    return old_ebic_select(S, r, n, penalty=penalty), w




def old_lpd(Xobs, y, M, penalty="lasso", eps_mode="relative", buggy=False):
    """Manuscript's non-robust LPD (unweighted IPW + LPD + EBIC), defects fixed."""
    return old_rlpd(Xobs, y, M, penalty, eps_mode, buggy, robust=False)
