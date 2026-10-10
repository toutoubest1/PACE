"""PACE estimators for sparse regression with missing covariates.

``full`` fits nuisance quantities and coefficients on the same sample.
``split`` fits nuisances on one half and coefficients on the other.
``crossfit`` fits nuisances outside each fold and pools the fold moments.
The fold count is controlled by ``cf_folds``; ``fit_pace`` uses five folds.
"""
import numpy as np
from dataclasses import dataclass
from .core import (robust_loc_scale, lpd, eig_clip, lpd_factor, hajek_moments, hajek_distance,
                   leverage_calibrate, leverage_weight_apply, bisq_w, pattern_residual_scale,
                   resid_cond, resid_zero)
from .solver import fit_path_select, fit_path_cv, lasso_path


@dataclass
class Nuisance:
    cell: str
    tau: float
    mu0: np.ndarray      # first-pass centre/scale (used for cell flagging)
    sd0: np.ndarray
    mu: np.ndarray       # centre/scale on usable cells
    sd: np.ndarray
    pi: np.ndarray       # column usable rates for the distance
    p: int
    Lmed: float          # leverage calibration
    s0sq: float
    beta: np.ndarray     # residual model
    Sp: np.ndarray       # working covariance for conditional imputation
    ybar: float
    rmed: float
    s_eps: float


# ------------------------------------------------------------------ preparation
def prepare_cells(Xobs, M, cell="flag", tau=3.5, mu0=None, sd0=None, mu=None, sd=None):
    """Usable-cell indicator U and centred cells Xc.  If (mu0, sd0, mu, sd) are supplied they
    are used as-is (cross-fitting); otherwise they are estimated from these rows."""
    U = M.copy()
    if mu0 is None:
        mu0, sd0 = robust_loc_scale(Xobs, U)
    Z = np.where(U, (Xobs - mu0) / sd0, 0.0)
    if cell == "flag":
        U = U & ~(np.abs(Z) > tau)
        if mu is None:
            mu, sd = robust_loc_scale(Xobs, U)
        Z = np.where(U, (Xobs - mu) / sd, 0.0)
    else:
        if mu is None:
            mu, sd = mu0, sd0
        Z = np.where(U, (Xobs - mu) / sd, 0.0)
        if cell == "winsor":
            Z = np.where(U, np.clip(Z, -tau, tau), 0.0)
    return U, Z, Z * sd, mu0, sd0, mu, sd


def final_fit(Xc, U, y, omega, penalty="mcp", stab="clip", one_se=True, folds=5, seed=0):
    """Weighted pairwise-Hajek moments + penalised quadratic loss, lambda by moment-CV."""
    n = len(y)
    ybar = np.sum(omega * y) / omega.sum()
    yc = y - ybar
    mf = lambda idx: hajek_moments(Xc[idx], yc[idx], U[idx], omega[idx])
    if stab == "cvalpha":
        best = None
        for f in (1.0, 0.75, 0.5, 0.25):
            bb, _ = fit_path_cv(mf, n, folds, penalty=penalty, lpd_fn=lpd_factor(f), one_se=one_se, seed=seed)
            if best is None or fit_path_cv.last_loss < best[0]:
                best = (fit_path_cv.last_loss, bb)
        return best[1]
    fn = eig_clip if stab == "clip" else lpd
    b, _ = fit_path_cv(mf, n, folds, penalty=penalty, lpd_fn=fn, one_se=one_se, seed=seed)
    return b


# ------------------------------------------------------------------ robust pilots
def robust_screen(Xc, U, y, wx, K):
    """Robust marginal screening: weighted Spearman correlation between y and each column
    over its usable rows (ranks bound the influence of vertical outliers)."""
    n, p = Xc.shape
    ry = (np.argsort(np.argsort(y)) + 0.5) / n - 0.5
    score = np.zeros(p)
    for j in range(p):
        idx = np.flatnonzero(U[:, j])
        if idx.size < 10:
            continue
        xr = (np.argsort(np.argsort(Xc[idx, j])) + 0.5) / idx.size - 0.5
        w = wx[idx]; yr = ry[idx] - np.sum(w * ry[idx]) / w.sum()
        xr = xr - np.sum(w * xr) / w.sum()
        score[j] = np.sum(w * xr * yr) / np.sqrt(np.sum(w * xr ** 2) * np.sum(w * yr ** 2) + 1e-12)
    return np.argsort(-np.abs(score))[:K], score


def pace_fit(Xobs, y, M, penalty="mcp", cell="flag", tau=3.5, use_wx=True, use_wy=True,
             wx_mode="hajek", stab="clip_final", pilot="cstep", n_csteps=3, n_irls=3, trim=0.75,
             k_bisq=4.685, resid="cond", screen_K=None, oracle_beta=None, return_nuisance=False,
             return_info=False):
    """Full-sample PACE estimator.

    stab : "clip_final" (final method: LPD inside pilot/IRLS, eigenvalue clipping at final fit),
           "lpd" (LPD everywhere), "clip" (clipping everywhere), "cvalpha" (LPD with intensity
           chosen by downstream moment-CV risk; ablation).
    pilot: "cstep"  (final: concentration steps, MCP + 1-SE moment-CV on trimmed rows, starts
                     'all' and 'zero', chosen by the pattern residual scale),
           "screen" / "screen+cstep" / "cstep3" (pilot variants examined in validation round 2).
    """
    n, p = Xobs.shape
    RES = resid_cond if resid == "cond" else resid_zero
    inner = eig_clip if stab == "clip" else lpd
    U, Z, Xc, mu0, sd0, mu, sd = prepare_cells(Xobs, M, cell, tau)
    pi = np.maximum(U.mean(0), 1.0 / n)
    d2, meff = hajek_distance(Z, U, pi)
    Lmed, s0sq = leverage_calibrate(d2, meff, p)
    if not use_wx:
        wx = np.ones(n)
    elif wx_mode == "hajek":
        wx, _ = leverage_weight_apply(d2, meff, p, Lmed, s0sq)
    elif wx_mode == "zf":
        d2zf = np.sum(np.where(U, Z, 0.0) ** 2, axis=1) / p
        wx, _ = leverage_weight_apply(d2zf, np.full(n, float(p)), p, *leverage_calibrate(d2zf, np.full(n, float(p)), p))
    elif wx_mode == "old":
        rn = np.linalg.norm(Xobs, axis=1); med = np.median(rn); mad = np.median(np.abs(rn - med))
        c = max(1.345 * mad / 0.6745 if mad > 1e-8 else 3.0, 1.0)
        wx = np.minimum(1.0, c / (rn + 1e-8))
    final_stab = {"clip_final": "clip", "clip": "clip", "lpd": "lpd", "cvalpha": "cvalpha"}[stab]

    def fit(omega, pen="lasso", final=False):
        ybar = np.sum(omega * y) / omega.sum() if omega.sum() > 0 else np.median(y)
        yc = y - ybar
        S, r, syy = hajek_moments(Xc, yc, U, omega)
        Sp, _ = inner(S)
        if final:
            b = final_fit(Xc, U, y, omega, penalty=pen, stab=final_stab)
        else:
            b, _ = fit_path_select(Sp, r, syy, n, penalty=pen, gamma_ebic=1.0)
        return b, yc, Sp, ybar

    if not use_wy:
        b, yc, Sp, ybar = fit(wx, penalty, final=True)
        return _ret(b, return_nuisance, return_info, None, dict(wx=wx, wy=np.ones(n), U=U))

    # ---------------- robust pilot
    def fit_trim(keep, one_se=True):
        idx = np.flatnonzero(keep)
        om = wx[idx]
        ybar = np.sum(om * y[idx]) / om.sum()
        yc_all = y - ybar
        Xi, Ui, yi = Xc[idx], U[idx], yc_all[idx]
        mf = lambda j: hajek_moments(Xi[j], yi[j], Ui[j], om[j])
        b, _ = fit_path_cv(mf, len(idx), 5, penalty="mcp", lpd_fn=inner, one_se=one_se, n_lams=25)
        S, r, _ = hajek_moments(Xi, yi, Ui, om)
        Sp, _ = inner(S)
        return b, yc_all, Sp

    h = int(trim * n)

    def csteps(b, yc, Sp):
        for _ in range(n_csteps):
            res, q = RES(Xc, U, yc, b, Sp)
            res = res - np.median(res)
            s = pattern_residual_scale(res, q)
            t2 = res ** 2 / (s ** 2 + q)
            keep = np.zeros(n, bool); keep[np.argsort(t2)[:h]] = True
            b_new, yc, Sp = fit_trim(keep)
            if np.max(np.abs(b_new - b)) < 1e-6:
                b = b_new; break
            b = b_new
        res, q = RES(Xc, U, yc, b, Sp); res -= np.median(res)
        return pattern_residual_scale(res, q), b, yc, Sp

    starts = []
    if pilot in ("cstep", "screen+cstep", "screen_cstep+cstep"):
        starts += ["all", "zero"]
    if pilot in ("screen", "screen+cstep", "screen_cstep", "screen_cstep+cstep"):
        starts += ["screen"]
    best = None
    S0 = Sp0 = None
    for st in starts:
        if st == "all":
            b, yc, Sp = fit_trim(np.ones(n, bool))
            cand = csteps(b, yc, Sp)
        elif st == "zero":
            cand = csteps(np.zeros(p), y - np.median(y), np.eye(p))
        else:  # two-stage screened robust pilot
            if Sp0 is None:
                yc0 = y - np.sum(wx * y) / wx.sum()
                S0, _, _ = hajek_moments(Xc, yc0, U, wx); Sp0, _ = inner(S0)
            K = screen_K or max(5, int(n / (2 * np.log(n))))
            Ks, _ = robust_screen(Xc, U, y, wx, K)
            b = _screened_lts(Xc, U, y, wx, Ks, p, h, RES, Sp0)
            yc = y - np.sum(wx * y) / wx.sum()
            if pilot in ("screen_cstep", "screen_cstep+cstep"):
                cand = csteps(b, yc, Sp0)
            else:
                res, q = RES(Xc, U, yc, b, Sp0); res -= np.median(res)
                cand = (pattern_residual_scale(res, q), b, yc, Sp0)
        if best is None or cand[0] < best[0]:
            best = cand
    if oracle_beta is not None:   # diagnostic only: true coefficients as pilot
        yc = y - np.sum(wx * y) / wx.sum()
        S0, _, _ = hajek_moments(Xc, yc, U, wx); Sp, _ = inner(S0)
        best = (None, oracle_beta.copy(), yc, Sp)
    s, b, yc, Sp = best
    b_pilot = b.copy()

    # ---------------- bisquare IRLS
    for it in range(n_irls):
        res, q = RES(Xc, U, yc, b, Sp); res -= np.median(res)
        if it == 0:
            s = pattern_residual_scale(res, q)
        t = res / np.sqrt(s ** 2 + q)
        wy = bisq_w(t, k_bisq)
        last = it == n_irls - 1
        b, yc, Sp, ybar = fit(wx * wy, penalty if last else "lasso", final=last)

    nuis = None
    if return_nuisance:
        res, q = RES(Xc, U, yc, b, Sp)
        rmed = float(np.median(res))
        s_e = pattern_residual_scale(res - rmed, q)
        nuis = Nuisance(cell, tau, mu0, sd0, mu, sd, pi, p, Lmed, s0sq, b.copy(), Sp, ybar, rmed, s_e)
    return _ret(b, return_nuisance, return_info, nuis, dict(wx=wx, wy=wy, U=U, b_pilot=b_pilot))


def _ret(b, rn, ri, nuis, info):
    if rn and ri:
        return b, nuis, info
    if rn:
        return b, nuis
    if ri:
        return b, info
    return b


def _screened_lts(Xc, U, y, wx, Ks, p, h, RES, Sp0, n_steps=10, ridge=1e-3):
    """Unpenalised LTS-type fit on the screened set Ks using pairwise-Hajek moments,
    started from all rows; residuals are pattern standardised."""
    n = len(y)
    keep = np.ones(n, bool)
    b = np.zeros(p)
    for _ in range(n_steps):
        om = wx * keep
        ybar = np.sum(om * y) / om.sum()
        yc = y - ybar
        S, r, _ = hajek_moments(Xc[:, Ks], yc, U[:, Ks], om)
        S = 0.5 * (S + S.T) + ridge * np.trace(S) / len(Ks) * np.eye(len(Ks))
        bk = np.linalg.solve(S, r)
        b_new = np.zeros(p); b_new[Ks] = bk
        res, q = RES(Xc, U, y - np.sum(wx * y) / wx.sum(), b_new, Sp0)
        res -= np.median(res)
        s = pattern_residual_scale(res, q)
        t2 = res ** 2 / (s ** 2 + q)
        new_keep = np.zeros(n, bool); new_keep[np.argsort(t2)[:h]] = True
        b = b_new
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return b


# ------------------------------------------------------------------ cross-fitting
def apply_nuisance(Xobs, y, M, nu: Nuisance, k_bisq=4.685, resid="cond", local_scale=False):
    """Weights for new rows computed from a nuisance fitted on *other* rows.
    local_scale=True re-estimates the two scalar residual nuisances (centre and pattern scale) on the
    target rows from out-of-fold residuals; the in-fold scale of the other fold is optimistic
    (it is computed from in-sample residuals)."""
    n, p = Xobs.shape
    RES = resid_cond if resid == "cond" else resid_zero
    U, Z, Xc, *_ = prepare_cells(Xobs, M, nu.cell, nu.tau, nu.mu0, nu.sd0, nu.mu, nu.sd)
    d2, meff = hajek_distance(Z, U, nu.pi)
    wx, _ = leverage_weight_apply(d2, meff, p, nu.Lmed, nu.s0sq)
    res, q = RES(Xc, U, y - nu.ybar, nu.beta, nu.Sp)
    if local_scale:
        rmed = float(np.median(res)); s_e = pattern_residual_scale(res - rmed, q)
    else:
        rmed, s_e = nu.rmed, nu.s_eps
    t = (res - rmed) / np.sqrt(s_e ** 2 + q)
    wy = bisq_w(t, k_bisq)
    return Xc, U, wx * wy


def pace(Xobs, y, M, mode="full", penalty="mcp", seed=12345, local_scale=False, cf_folds=2, **kw):
    """Front end.  mode in {"full", "split", "crossfit"}."""
    n = len(y)
    if mode == "full":
        return pace_fit(Xobs, y, M, penalty=penalty, **kw)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    stab = kw.get("stab", "clip_final")
    final_stab = "clip" if stab in ("clip_final", "clip") else "lpd"
    if mode == "split":
        _, nuA = pace_fit(Xobs[A], y[A], M[A], penalty=penalty, return_nuisance=True, **kw)
        XcB, UB, omB = apply_nuisance(Xobs[B], y[B], M[B], nuA, local_scale=local_scale)
        return final_fit(XcB, UB, y[B], omB, penalty=penalty, stab=final_stab)
    if mode == "crossfit":
        folds = kw.pop("folds", 2) if "folds" in kw else cf_folds
        fid = np.empty(n, int); fid[perm] = np.arange(n) % folds
        Xc = np.zeros(Xobs.shape); U = np.zeros(Xobs.shape, bool); om = np.zeros(n)
        for k in range(folds):
            tr = np.flatnonzero(fid != k); te = np.flatnonzero(fid == k)
            _, nu = pace_fit(Xobs[tr], y[tr], M[tr], penalty=penalty, return_nuisance=True, **kw)
            Xc[te], U[te], om[te] = apply_nuisance(Xobs[te], y[te], M[te], nu, local_scale=local_scale)
        return final_fit(Xc, U, y, om, penalty=penalty, stab=final_stab)
    raise ValueError(mode)
