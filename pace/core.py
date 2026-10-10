"""Building blocks: robust standardisation, weighted pairwise-Hajek moments, pattern-adjusted
distance, leverage / residual weights, PSD repair, pattern-standardised residuals."""
import numpy as np
from numpy.linalg import eigvalsh

MAD_CONST = 0.6745


def robust_loc_scale(X, U):
    """Column-wise median and 1.4826*MAD over usable cells (U==True)."""
    Xn = np.where(U, X, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.nanmedian(Xn, axis=0)
            sd = 1.4826 * np.nanmedian(np.abs(Xn - mu), axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    return mu, sd


def lpd(S, eta=0.02):
    """Linear-shrinkage PD repair alpha*S + (1-alpha)*mu*I with relative floor eta*mu
    (positive-definiteness boundary)."""
    S = 0.5 * (S + S.T)
    p = S.shape[0]
    mu = np.trace(S) / p
    eps = eta * mu
    lmin = eigvalsh(S)[0]
    if lmin >= eps:
        return S, 1.0
    a = (mu - eps) / (mu - lmin)
    return a * S + (1 - a) * mu * np.eye(p), a


def eig_clip(S, eta=0.02):
    """PSD repair by eigenvalue clipping at eta * (mean eigenvalue)."""
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    floor = eta * max(np.mean(w), 1e-8)
    return (V * np.maximum(w, floor)) @ V.T, 1.0


def lpd_factor(f):
    """LPD with intensity f * alpha_boundary (used for the downstream-risk shrinkage ablation)."""
    def g(S):
        Sp, a = lpd(S)
        if a >= 1.0 or f == 1.0:
            return Sp, a
        p = S.shape[0]; mu = np.trace(S) / p
        a2 = f * a
        return a2 * S + (1 - a2) * mu * np.eye(p), a2
    return g


def hajek_moments(Xc, yc, U, omega):
    """Weighted pairwise-Hajek IPW moments
        Sigma_jk = sum_i w_i U_ij U_ik x_ij x_ik / sum_i w_i U_ij U_ik
        rho_j    = sum_i w_i U_ij x_ij y_i     / sum_i w_i U_ij
        s_yy     = sum_i w_i y_i^2 / sum_i w_i."""
    Uf = U.astype(float)
    Xu = np.where(U, Xc, 0.0)
    Xw = Xu * omega[:, None]
    A = Xw.T @ Xu
    D = (Uf * omega[:, None]).T @ Uf
    S = A / np.maximum(D, 1e-8 * omega.sum())
    rho = (Xw.T @ yc) / np.maximum(Uf.T @ omega, 1e-12)
    syy = np.sum(omega * yc ** 2) / omega.sum()
    return 0.5 * (S + S.T), rho, syy


def hajek_distance(Z, U, pi):
    """Pattern-adjusted (row-Hajek) distance d_i^2 = sum_j (U_ij/pi_j) z_ij^2 / sum_j (U_ij/pi_j)
    and the effective number of observed coordinates m_i."""
    Wt = U / pi[None, :]
    num = np.sum(Wt * np.where(U, Z, 0.0) ** 2, axis=1)
    den = np.maximum(Wt.sum(axis=1), 1e-12)
    m_eff = den ** 2 / np.maximum(np.sum(Wt ** 2, axis=1), 1e-12)
    return num / den, m_eff


def leverage_calibrate(d2, m_eff, p, kappa=2.0):
    """Centre/scale of L_i = log d_i^2 with the pattern-variance correction removed."""
    L = np.log(np.maximum(d2, 1e-12))
    med = np.median(L)
    v = kappa * np.maximum(1.0 / m_eff - 1.0 / p, 0.0)
    mad2 = (1.4826 * np.median(np.abs(L - med))) ** 2
    s0sq = max(mad2 - np.median(v), 0.25 * mad2, 1e-4)
    return med, s0sq


def leverage_weight_apply(d2, m_eff, p, med, s0sq, c1=2.5, c2=5.0, kappa=2.0):
    """One-sided redescending (bisquare-taper) weight of the pattern-standardised log distance."""
    L = np.log(np.maximum(d2, 1e-12))
    v = kappa * np.maximum(1.0 / m_eff - 1.0 / p, 0.0)
    t = (L - med) / np.sqrt(s0sq + v)
    w = np.ones_like(t)
    mid = (t > c1) & (t < c2)
    w[mid] = (1 - ((t[mid] - c1) / (c2 - c1)) ** 2) ** 2
    w[t >= c2] = 0.0
    return w, t


def leverage_weight(d2, m_eff, p, **kw):
    med, s0sq = leverage_calibrate(d2, m_eff, p)
    return leverage_weight_apply(d2, m_eff, p, med, s0sq, **kw)


def bisq_w(t, k=4.685):
    a = np.abs(t) / k
    return np.where(a < 1, (1 - a ** 2) ** 2, 0.0)


def pattern_residual_scale(r, q):
    """Solve median_i |r_i| / sqrt(s^2 + q_i) = 0.6745 for s (bisection)."""
    lo, hi = 1e-6, 10 * (np.median(np.abs(r)) / MAD_CONST + 1e-6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.median(np.abs(r) / np.sqrt(mid ** 2 + q)) > MAD_CONST:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def resid_cond(Xc, U, yc, b, Sp, k_nb=8):
    """Pattern-standardised residual ingredients.
    Missing (or flagged) active covariates are replaced by their Gaussian conditional mean given
    observed covariates in a local neighbourhood (working covariance Sp);
    q_i = beta_m' Var(x_m | x_o) beta_m is the row-specific residual-variance inflation."""
    n = len(yc)
    nz = np.flatnonzero(b)
    Xu = np.where(U, Xc, 0.0)
    res = yc - Xu @ b
    q = np.zeros(n)
    if nz.size == 0:
        return res, q
    nb = set(nz.tolist())
    for j in nz:
        nb.update(np.argsort(-np.abs(Sp[j]))[: k_nb + 1].tolist())
    nb = np.array(sorted(nb))
    for i in range(n):
        mis = nz[~U[i, nz]]
        if mis.size == 0:
            continue
        obs = nb[U[i, nb]]
        bm = b[mis]
        if obs.size:
            Soo = Sp[np.ix_(obs, obs)]; Som = Sp[np.ix_(obs, mis)]
            A = np.linalg.solve(Soo, Som)
            xm_hat = Xc[i, obs] @ A
            Vc = Sp[np.ix_(mis, mis)] - Som.T @ A
        else:
            xm_hat = np.zeros(mis.size); Vc = Sp[np.ix_(mis, mis)]
        res[i] -= xm_hat @ bm
        q[i] = max(bm @ Vc @ bm, 0.0)
    return res, q


def resid_zero(Xc, U, yc, b, Sp):
    """Zero-fill residual with marginal variance inflation (ablation only)."""
    Xu = np.where(U, Xc, 0.0)
    res = yc - Xu @ b
    nz = np.flatnonzero(b)
    if nz.size:
        Bm = np.where(~U[:, nz], 1.0, 0.0) * b[nz][None, :]
        q = np.einsum("ij,jk,ik->i", Bm, Sp[np.ix_(nz, nz)], Bm)
    else:
        q = np.zeros(len(yc))
    return res, q
