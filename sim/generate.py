"""Data-generating mechanisms for all simulation regimes and evaluation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def ar1(p, rho=0.5):
    return rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))


def gen(n=200, p=300, s=10, seed=0, rho=0.5, err="normal", contam="none", eps=0.1,
        miss="mcar", mrate=0.2, cell_rate=0.02, bl=1.0, bh=2.0, xshift=3.0, yshift=15.0):
    rng = np.random.default_rng(seed)
    Sig = ar1(p, rho)
    L = np.linalg.cholesky(Sig)
    X = rng.standard_normal((n, p)) @ L.T
    beta = np.zeros(p)
    S = rng.choice(p, s, replace=False)
    beta[S] = rng.choice([-1, 1], s) * rng.uniform(bl, bh, s)
    if err == "normal":
        e = rng.standard_normal(n)
    elif err == "t2":
        e = rng.standard_t(2, n)
    elif err == "cauchy":
        e = rng.standard_cauchy(n)
    y = X @ beta + e
    Xc = X.copy(); yc = y.copy()
    m = int(round(eps * n))
    out_rows = rng.choice(n, m, replace=False)
    J = rng.choice(p, p // 2, replace=False)  # coordinated shift direction
    if contam in ("x", "xy"):
        Xc[np.ix_(out_rows, J)] += xshift
    if contam in ("y", "xy"):
        yc[out_rows] += yshift
    if contam == "xlev":  # gross leverage: large spherical inflation (old paper's mechanism)
        dirn = rng.standard_normal((m, p)); dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
        Xc[out_rows] += 8 * dirn * np.linalg.norm(X[out_rows], axis=1, keepdims=True)
    if contam in ("cell", "cell4"):
        C = rng.random((n, p)) < cell_rate
        Xc[C] += rng.choice([-1, 1], C.sum()) * (6.0 if contam == "cell" else 4.0)
    # missingness
    if miss == "mcar":
        P = np.full((n, p), mrate)
    elif miss == "hetcol":  # heterogeneous column probabilities, mean = mrate
        pj = rng.uniform(0.2 * mrate, 1.8 * mrate, p)
        P = np.tile(pj, (n, 1))
    elif miss == "hetrow":  # half of the rows nearly complete, half heavily missing (MCAR)
        rr = np.where(rng.random(n) < 0.5, 0.25 * mrate, 1.75 * mrate)
        P = np.tile(rr[:, None], (1, p))
    elif miss == "mar":  # driven by an always-observed column 0 (MAR)
        z = X[:, 0]
        a = np.log(mrate / (1 - mrate))
        P = np.tile((1 / (1 + np.exp(-(a + 1.0 * z))))[:, None], (1, p)); P[:, 0] = 0
    elif miss == "mnar":  # self-masking (old paper's mechanism)
        q75 = np.quantile(X, 0.75, axis=0)
        P = np.where(X > q75, min(2 * mrate, 0.95), 0.5 * mrate)
    Mobs = rng.random((n, p)) >= P
    Xobs = np.where(Mobs, Xc, 0.0)
    return dict(Xobs=Xobs, y=yc, M=Mobs, beta=beta, Sigma=Sig, out_rows=out_rows)


def metrics(b, beta, Sig, thr=1e-6):
    sel = np.abs(b) > thr
    tru = beta != 0
    tp = np.sum(sel & tru); fp = np.sum(sel & ~tru)
    d = b - beta
    try:
        auc = roc_auc_score(tru.astype(int), np.abs(b))
    except Exception:
        auc = 0.5
    prec = tp / max(sel.sum(), 1); rec = tp / tru.sum()
    f1 = 0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return dict(L2=np.linalg.norm(d), PE=float(d @ Sig @ d), AUC=auc, F1=f1, TPR=rec,
                FDR=fp / max(sel.sum(), 1), Size=int(sel.sum()))


