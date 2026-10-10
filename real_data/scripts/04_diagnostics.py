"""Full-data PACE-CF5 diagnostics per drug and the predictor-only pattern-distortion diagnostic
(protocol Section 7).  Usage: python scripts/04_diagnostics.py [--debug]
Outputs (results/raw/ or results/debug/):
  diag_weights_<drug>.csv   per-line weights (w_x, w_y, omega), row missing rate, batch, lineage
  diag_cells_<drug>.json    flagged-cell fraction, selected proteins, summaries
  diag_pattern_lines.csv    per-line leverage measures (all 375 lines, predictors only)
  diag_pattern_summary.csv  Spearman / partial Spearman / tertile summaries"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PANEL, load, drug_data, standardiser, to_input, fit_pace  # noqa: E402
from pace.estimator import pace_fit, apply_nuisance, final_fit, prepare_cells   # noqa: E402
from pace.core import hajek_distance, leverage_calibrate, leverage_weight_apply   # noqa: E402

debug = "--debug" in sys.argv
odir = os.path.join(ROOT, "results", "debug" if debug else "raw")
os.makedirs(odir, exist_ok=True)
X, A, L = load()
prot = np.array(X.columns)


def cf5_with_weights(Xobs, y, M, seed=12345, folds=5):
    """Exactly the steps of pace(mode='crossfit', cf_folds=5) (= fit_pace), additionally returning
    the leverage weight w_x, the residual weight w_y and the usable-cell indicator U of every row."""
    n, p = Xobs.shape
    perm = np.random.default_rng(seed).permutation(n)
    fid = np.empty(n, int); fid[perm] = np.arange(n) % folds
    Xc = np.zeros(Xobs.shape); U = np.zeros(Xobs.shape, bool); om = np.zeros(n); wx = np.zeros(n)
    for k in range(folds):
        tr = np.flatnonzero(fid != k); te = np.flatnonzero(fid == k)
        _, nu = pace_fit(Xobs[tr], y[tr], M[tr], penalty="mcp", return_nuisance=True)
        Xc[te], U[te], om[te] = apply_nuisance(Xobs[te], y[te], M[te], nu)
        Ut, Zt, *_ = prepare_cells(Xobs[te], M[te], nu.cell, nu.tau, nu.mu0, nu.sd0, nu.mu, nu.sd)
        d2, meff = hajek_distance(Zt, Ut, nu.pi)
        wx[te], _ = leverage_weight_apply(d2, meff, p, nu.Lmed, nu.s0sq)
    b = final_fit(Xc, U, y, om, penalty="mcp", stab="clip")
    wy = np.where(wx > 0, om / np.maximum(wx, 1e-300), np.nan)
    return b, wx, wy, om, U


# ------------------------------------------------------------ per-drug full-data fits
for drug in PANEL:
    Xnan, y, lines = drug_data(X, A, drug)
    if debug:
        y = np.random.default_rng(999).standard_normal(len(y))
    med, sc = standardiser(Xnan)
    Z, M = to_input(Xnan, med, sc)
    yc = y - np.median(y)
    b, wx, wy, om, U = cf5_with_weights(Z, yc, M)
    b_ref = fit_pace(Z, yc, M)
    assert np.array_equal(b, b_ref), f"{drug}: diagnostic re-run differs from fit_pace"
    rm = 1 - M.mean(axis=1)
    W = pd.DataFrame(dict(line=lines, wx=wx, wy=wy, omega=om, row_missing=rm,
                          batch=L.loc[lines, "PROTEOMICS_10PLEX_ID"].values,
                          lineage=L.loc[lines, "LINEAGE"].values, y=y))
    W.to_csv(os.path.join(odir, f"diag_weights_{drug}.csv"), index=False)
    flagged = (M & ~U)
    sel = np.flatnonzero(b)
    low = W.omega < 0.5
    info = dict(drug=drug, n=len(y), p=int(Z.shape[1]),
                frac_cells_flagged=float(flagged.sum() / M.sum()),
                frac_lines_any_flag=float(flagged.any(axis=1).mean()),
                frac_omega_lt_05=float(low.mean()), frac_omega_eq_0=float((W.omega == 0).mean()),
                frac_wx_lt_1=float((W.wx < 1).mean()), frac_wy_lt_05=float((W.wy < 0.5).mean()),
                spearman_omega_rowmissing=float(spearmanr(W.omega, rm).statistic),
                spearman_wx_rowmissing=float(spearmanr(W.wx, rm).statistic),
                row_missing_mean_all=float(rm.mean()),
                row_missing_mean_low=float(rm[low.values].mean()) if low.any() else None,
                lineage_low=W.loc[low, "lineage"].value_counts().to_dict(),
                lineage_all=W["lineage"].value_counts().to_dict(),
                batches_low=int(W.loc[low, "batch"].nunique()),
                y_low=W.loc[low, "y"].round(3).tolist(),
                selected=[dict(protein=str(prot[j]), coef=float(b[j])) for j in sel],
                flagged_cells_per_protein_top=pd.Series(flagged.sum(axis=0), index=prot)
                .sort_values(ascending=False).head(10).to_dict())
    with open(os.path.join(odir, f"diag_cells_{drug}.json"), "w") as f:
        json.dump(info, f, indent=1, default=int)
    print(drug, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()
                 if k in ("n", "frac_cells_flagged", "frac_omega_lt_05", "spearman_omega_rowmissing")}, "size", len(sel))

# ------------------------------------------------------------ pattern-distortion diagnostic (predictors only)
Xall = X.values.astype(float)
med, sc = standardiser(Xall)
Z, M = to_input(Xall, med, sc)
n, p = Z.shape
rm = 1 - M.mean(axis=1)
comp = M.all(axis=0)
ref = np.sum(Z[:, comp] ** 2, axis=1) / comp.sum()                     # (c) pattern-free reference
zf = np.sum(Z ** 2, axis=1) / p                                          # (a) zero-filled norm
rn = np.linalg.norm(Z, axis=1)                                           # (a) radial weight of RLPD
mad_rn = np.median(np.abs(rn - np.median(rn)))
c = max(1.345 * mad_rn / 0.6745 if mad_rn > 1e-8 else 3.0, 1.0)
w_rad = np.minimum(1.0, c / (rn + 1e-8))
Ub, Zb, *_ = prepare_cells(Z, M, "flag", 3.5)                           # (b) PACE distance and weight
pi = np.maximum(Ub.mean(0), 1.0 / n)
d2, meff = hajek_distance(Zb, Ub, pi)
Lmed, s0sq = leverage_calibrate(d2, meff, p)
w_pace, _ = leverage_weight_apply(d2, meff, p, Lmed, s0sq)
d2zf = np.sum(np.where(Ub, Zb, 0.0) ** 2, axis=1) / p                    # zero-filled distance, PACE weight fn
w_zf, _ = leverage_weight_apply(d2zf, np.full(n, float(p)), p, *leverage_calibrate(d2zf, np.full(n, float(p)), p))
D = pd.DataFrame(dict(line=X.index, batch=L.loc[X.index, "PROTEOMICS_10PLEX_ID"].values,
                      lineage=L.loc[X.index, "LINEAGE"].values, row_missing=rm, ref=ref,
                      zf_norm=zf, radial_w=w_rad, zf_pacefn_w=w_zf, pace_d2=d2, pace_wx=w_pace))
D.to_csv(os.path.join(odir, "diag_pattern_lines.csv"), index=False)


def partial_spearman(a, b, ctrl):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(ctrl)
    Xc = np.c_[np.ones(len(rc)), rc]
    res = lambda v: v - Xc @ np.linalg.lstsq(Xc, v, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


ter = pd.qcut(rm, 3, labels=["low", "mid", "high"])
rows = []
for name, v, down in [("zero-filled norm ||z_obs||^2/p", zf, None), ("radial weight (RLPD)", w_rad, w_rad < 1),
                      ("zero-filled distance + PACE weight fn", w_zf, w_zf < 1),
                      ("PACE distance d^2", d2, None), ("PACE leverage weight w_x", w_pace, w_pace < 1),
                      ("reference: complete proteins", ref, None)]:
    r = dict(measure=name, spearman_rowmissing=float(spearmanr(v, rm).statistic),
             partial_spearman_given_ref=partial_spearman(v, rm, ref) if name.startswith("reference") is False else np.nan,
             spearman_with_ref=float(spearmanr(v, ref).statistic))
    for t in ["low", "mid", "high"]:
        idx = np.asarray(ter == t)
        r[f"mean_{t}"] = float(np.mean(v[idx]))
        if down is not None:
            r[f"share_down_{t}"] = float(np.mean(down[idx]))
    rows.append(r)
S = pd.DataFrame(rows)
S.attrs["n_complete_proteins"] = int(comp.sum())
S.to_csv(os.path.join(odir, "diag_pattern_summary.csv"), index=False)
print("complete proteins for reference:", int(comp.sum()))
print(S.round(3).to_string())
