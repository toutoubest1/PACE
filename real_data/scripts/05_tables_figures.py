"""Tables (LaTeX + CSV) and figures of the real-data application from results/raw.
Usage: python scripts/05_tables_figures.py [--debug]"""
import os, sys, json, glob, itertools, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PANEL, ORDER, METHODS  # noqa: E402

debug = "--debug" in sys.argv
RAW = os.path.join(ROOT, "results", "debug" if debug else "raw")
TAB = os.path.join(ROOT, "results", "debug_tables" if debug else "tables")
FIG = os.path.join(ROOT, "results", "debug_figures" if debug else "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a96"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42})
DRUGLAB = {"Daporinad": "Daporinad", "Gemcitabine": "Gemcitabine", "THZ-2-102-1": "THZ-2-102-1",
           "Bleomycin": "Bleomycin", "Trametinib": "Trametinib", "SNX-2112": "SNX-2112"}
TEXNAME = {m: m for m in ORDER}
TEXNAME["Null"] = "Null (training median)"

files = sorted(f for f in glob.glob(os.path.join(RAW, "*_r[0-9][0-9].csv")))
R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# Spearman correlation is undefined for a constant prediction (empty model); it is set to 0
# (no ranking information).  Deviation log, REAL_DATA_PROTOCOL.md Section 10, item 1.
for c in ("Spearman", "Spearman_mice"):
    R["const_pred" + c[8:]] = R[c].isna() & (R.Method != "Null")
    R.loc[R.Method != "Null", c] = R.loc[R.Method != "Null", c].fillna(0.0)
R.to_csv(os.path.join(TAB, "all_splits_long.csv"), index=False)
drugs = [d for d in PANEL if d in set(R.drug)]
nrep = R.groupby("drug").rep.nunique().to_dict()


def mse(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def fmt(m, s, d=3):
    return f"{m:.{d}f}" if not np.isfinite(s) else f"{m:.{d}f} ({s:.{d}f})"


def tex_table(df, caption, label, colfmt=None, fname=None, bold_min_cols=None, note=None):
    cols = [c for c in df.columns if not str(c).startswith("_")]
    colfmt = colfmt or "l" + "r" * (len(cols) - 1)
    lines = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", "\\scriptsize",
             "\\fitbox{\\begin{tabular}{" + colfmt + "}", "\\toprule",
             " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = str(r[c])
            if bold_min_cols and c in bold_min_cols and r.get("_best_" + str(c), False):
                v = "\\textbf{" + v + "}"
            cells.append(v)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    if note:
        lines += ["\\par\\smallskip\\parbox{\\linewidth}{\\scriptsize " + note + "}"]
    lines += ["\\end{table}"]
    with open(os.path.join(TAB, fname), "w") as f:
        f.write("\n".join(lines) + "\n")


def metric_table(metric, fname, caption, label, d=3, note=None):
    rows, best = [], {}
    for dr in drugs:
        g = R[R.drug == dr].groupby("Method")[metric]
        best[dr] = g.mean().drop("Null").idxmin() if metric not in ("Spearman", "Spearman_mice") else g.mean().drop("Null").idxmax()
    ranks = []
    for dr in drugs:
        g = R[(R.drug == dr) & (R.Method != "Null")].groupby("Method")[metric].mean()
        ranks.append(g.rank(ascending=metric not in ("Spearman", "Spearman_mice")))
    mean_rank = pd.concat(ranks, axis=1).mean(axis=1)
    for m in ORDER:
        r = {"Method": TEXNAME[m]}
        for dr in drugs:
            v = R[(R.drug == dr) & (R.Method == m)][metric]
            r[DRUGLAB[dr]] = fmt(*mse(v), d)
            r["_best_" + DRUGLAB[dr]] = (best[dr] == m)
        r["Mean rank"] = f"{mean_rank[m]:.1f}" if m in mean_rank else "--"
        rows.append(r)
    df = pd.DataFrame(rows)
    df[[c for c in df.columns if not c.startswith("_")]].to_csv(os.path.join(TAB, fname.replace(".tex", ".csv")), index=False)
    tex_table(df[[c for c in df.columns]], caption, label, fname=fname,
              bold_min_cols=[DRUGLAB[d] for d in drugs], note=note)
    df = df[[c for c in df.columns if not c.startswith("_")]]
    return df, mean_rank


n_txt = ", ".join(f"{DRUGLAB[d]} {int(R[(R.drug == d)].n_train.iloc[0] + R[(R.drug == d)].n_test.iloc[0])}" for d in drugs)
R_used = min(nrep.values())
common_note = (f"Mean (standard error) over {R_used} random 80/20 splits per drug; number of lines: {n_txt}; "
               "$p=500$ proteins. Bold: best method for the drug (Null excluded). Splits overlap, so standard errors are descriptive.")
mae, rank_mae = metric_table("MAE", "rd_mae.tex", "Held-out mean absolute error of drug-response AUC (primary metric; "
                             "common prediction rule: training-median imputation of missing test proteins and median-residual intercept).",
                             "tab:rd-mae", note=common_note)
metric_table("RMSE", "rd_rmse.tex", "Held-out root mean squared error (common prediction rule).", "tab:rd-rmse", note=common_note)
metric_table("Spearman", "rd_spearman.tex", "Held-out Spearman correlation between predicted and observed AUC (common prediction rule).",
             "tab:rd-spearman", note=common_note.replace("Bold: best", "Bold: largest"))
metric_table("MAE_mice", "rd_mae_mice.tex", "Held-out mean absolute error under the secondary prediction rule (missing test proteins "
             "imputed by an iterative imputer fitted on the training lines).", "tab:rd-mae-mice", note=common_note)
metric_table("Size", "rd_size.tex", "Selected model size $\\|\\hat\\beta\\|_0$.", "tab:rd-size", d=1,
             note=common_note.replace(" Bold: best method for the drug (Null excluded).", ""))
metric_table("Time", "rd_time.tex", "Computing time in seconds (single core, process time).", "tab:rd-time", d=1,
             note=common_note.replace(" Bold: best method for the drug (Null excluded).", ""))

# ---------------------------------------------------------------- paired differences vs PACE-CF5 (MAE)
rows = []
for m in ORDER:
    if m == "PACE-CF5":
        continue
    r = {"Method": TEXNAME[m]}
    wins = 0; tot = 0
    for dr in drugs:
        a = R[(R.drug == dr) & (R.Method == "PACE-CF5")].set_index("rep").MAE
        b = R[(R.drug == dr) & (R.Method == m)].set_index("rep").MAE
        dd = (b - a).dropna()
        r[DRUGLAB[dr]] = fmt(*mse(dd), 3)
        wins += int((dd < 0).sum()); tot += len(dd)
    r["Splits better than PACE-CF5"] = f"{wins}/{tot}"
    rows.append(r)
P = pd.DataFrame(rows)
P.to_csv(os.path.join(TAB, "rd_paired.csv"), index=False)
tex_table(P, "Paired difference in held-out MAE, method minus PACE-CF5 (positive: PACE-CF5 more accurate), "
          "and the number of splits in which the method had a smaller MAE than PACE-CF5.", "tab:rd-paired",
          fname="rd_paired.tex", note=common_note.replace(" Bold: best method for the drug (Null excluded).", ""))

# ---------------------------------------------------------------- stability
coef = {}
for f in sorted(glob.glob(os.path.join(RAW, "*_r[0-9][0-9]_coef.npz"))):
    base = os.path.basename(f)[:-9]
    dr, rep = base.rsplit("_r", 1)
    z = np.load(f)
    for m in METHODS:
        coef.setdefault((dr, m), {})[int(rep)] = z[m.replace(" ", "_")]


def jacc(a, b):
    A, B = set(np.flatnonzero(a)), set(np.flatnonzero(b))
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


st_rows = []
for dr in drugs:
    for m in METHODS:
        bs = [coef[(dr, m)][k] for k in sorted(coef[(dr, m)])]
        pairs = list(itertools.combinations(range(len(bs)), 2))
        J = np.mean([jacc(bs[i], bs[j]) for i, j in pairs]) if pairs else np.nan
        Cs = np.mean([cosine(bs[i], bs[j]) for i, j in pairs]) if pairs else np.nan
        freq = np.mean([b != 0 for b in bs], axis=0)
        st_rows.append(dict(drug=dr, Method=m, jaccard=J, cosine=Cs, stable50=int((freq >= 0.5).sum()),
                            size=float(np.mean([np.sum(b != 0) for b in bs])),
                            empty=float(np.mean([np.sum(b != 0) == 0 for b in bs]))))
ST = pd.DataFrame(st_rows)
ST.to_csv(os.path.join(TAB, "rd_stability_long.csv"), index=False)
rows = []
for m in METHODS:
    s = ST[ST.Method == m]
    r = {"Method": m}
    for dr in drugs:
        x = s[s.drug == dr].iloc[0]
        r[DRUGLAB[dr]] = f"{x.jaccard:.2f} / {x.cosine:.2f} / {x.stable50:d}"
    r["Mean Jaccard"] = f"{s.jaccard.mean():.2f}"
    r["Mean cosine"] = f"{s.cosine.mean():.2f}"
    rows.append(r)
S2 = pd.DataFrame(rows)
S2.to_csv(os.path.join(TAB, "rd_stability.csv"), index=False)
tex_table(S2, "Selection and coefficient stability across the training fits: mean pairwise Jaccard index of the selected "
          "sets / mean pairwise cosine similarity of $\\hat\\beta$ / number of proteins selected in at least half of the splits.",
          "tab:rd-stability", fname="rd_stability.tex",
          note=f"{R_used} splits per drug. Pairs of empty models count as Jaccard 1 and cosine 0.")

# ---------------------------------------------------------------- compact main-paper summary table
rows = []
for m in ORDER:
    r = {"Method": TEXNAME[m]}
    sub = R[R.Method == m]
    rel = sub.groupby("drug").relMAE.mean()
    r["Rel.\\ MAE (range over drugs)"] = f"{rel.mean():.3f} ({rel.min():.3f}--{rel.max():.3f})"
    r["Mean rank MAE"] = f"{rank_mae[m]:.1f}" if m in rank_mae else "--"
    r["Spearman"] = f"{sub.groupby('drug').Spearman.mean().mean():.2f}" if m != "Null" else "--"
    r["Size"] = f"{sub.groupby('drug').Size.mean().mean():.1f}"
    r["Empty (\\%)"] = f"{100 * (sub.Size == 0).mean():.0f}" if m != "Null" else "--"
    if m in METHODS:
        s = ST[ST.Method == m]
        r["Jaccard"] = f"{s.jaccard.mean():.2f}"
        r["Cosine"] = f"{s.cosine.mean():.2f}"
        r["Time (s)"] = f"{sub.Time.mean():.1f}"
    else:
        r["Jaccard"] = r["Cosine"] = r["Time (s)"] = "--"
    rows.append(r)
SM = pd.DataFrame(rows)
SM.to_csv(os.path.join(TAB, "rd_summary.csv"), index=False)
tex_table(SM, "Real-data application, summary over the six drugs: held-out MAE relative to the Null predictor (mean over drugs, "
          "with range of the per-drug means), mean rank of MAE (1 = best of 11), held-out Spearman correlation, model size, "
          "selection stability (Jaccard), coefficient stability (cosine) and computing time.", "tab:rd-summary",
          fname="rd_summary.tex", note=f"{R_used} random 80/20 splits per drug, $p=500$ proteins. Per-drug results: Supplement~\\ref{{S-sec-rd}}.")

# ---------------------------------------------------------------- PACE-CF5 full-data diagnostics
drows = []
for dr in drugs:
    fj = os.path.join(RAW, f"diag_cells_{dr}.json")
    if not os.path.exists(fj):
        continue
    I = json.load(open(fj))
    W = pd.read_csv(os.path.join(RAW, f"diag_weights_{dr}.csv"))
    low = W.omega < 0.5
    top_lin = W.loc[low, "lineage"].value_counts()
    drows.append({"Drug": DRUGLAB[dr], "$n$": I["n"], "Cells flagged (\\%)": f"{100 * I['frac_cells_flagged']:.2f}",
                  "$w_x<1$ (\\%)": f"{100 * I['frac_wx_lt_1']:.1f}", "$w_y<0.5$ (\\%)": f"{100 * I['frac_wy_lt_05']:.1f}",
                  "$\\omega<0.5$ (\\%)": f"{100 * I['frac_omega_lt_05']:.1f}",
                  "$\\omega=0$ (\\%)": f"{100 * I['frac_omega_eq_0']:.1f}",
                  "$r_S(\\omega,$ miss.)": f"{I['spearman_omega_rowmissing']:.2f}",
                  "Miss.\\ rate: all / $\\omega<0.5$": f"{I['row_missing_mean_all']:.3f} / " +
                  (f"{I['row_missing_mean_low']:.3f}" if I["row_missing_mean_low"] is not None else "--"),
                  "Size": len(I["selected"]),
                  "Most common lineage, $\\omega<0.5$": (f"{top_lin.index[0].replace('_', ' ')} ({top_lin.iloc[0]}/{int(low.sum())})"
                                                          if low.any() else "--")})
if drows:
    DG = pd.DataFrame(drows)
    DG.to_csv(os.path.join(TAB, "rd_diag.csv"), index=False)
    tex_table(DG, "PACE-CF5 fitted to all lines of each drug: share of observed cells flagged, share of lines with reduced "
              "leverage weight, residual weight and final weight, Spearman correlation of the final weight with the row missing "
              "rate, mean row missing rate of all lines and of the strongly downweighted lines, model size, and the most common "
              "lineage among strongly downweighted lines.", "tab:rd-diag", fname="rd_diag.tex")
    with open(os.path.join(TAB, "rd_selected.tex"), "w") as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n\\caption{Proteins selected by PACE-CF5 on all lines of each drug "
                "(coefficient on the robustly standardised scale). Listed for transparency; no biological claim is made.}"
                "\\label{tab:rd-selected}\n\\scriptsize\n\\begin{tabular}{lp{0.72\\linewidth}}\n\\toprule\nDrug & Selected proteins \\\\\n\\midrule\n")
        for dr in drugs:
            I = json.load(open(os.path.join(RAW, f"diag_cells_{dr}.json")))
            s = sorted(I["selected"], key=lambda d: -abs(d["coef"]))
            txt = ", ".join(f"{d['protein'].split('|')[0]} ({d['coef']:+.2g})" for d in s) or "none"
            f.write(f"{DRUGLAB[dr]} & {txt.replace('_', chr(92) + '_')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- pattern-distortion diagnostic
fp = os.path.join(RAW, "diag_pattern_lines.csv")
if os.path.exists(fp):
    PL = pd.read_csv(fp)
    PS = pd.read_csv(os.path.join(RAW, "diag_pattern_summary.csv"))
    LAB = {"zero-filled norm ||z_obs||^2/p": "Zero-filled norm $\\|z_{\\mathrm{obs}}\\|^2/p$",
           "radial weight (RLPD)": "Radial weight $\\min(1,c/\\|x_{\\mathrm{obs}}\\|)$ (RLPD)",
           "zero-filled distance + PACE weight fn": "Zero-filled distance, PACE weight function",
           "PACE distance d^2": "PACE pattern-adjusted distance $d^2$",
           "PACE leverage weight w_x": "PACE leverage weight $w_x$",
           "reference: complete proteins": "Reference: proteins observed in all lines"}
    rows = []
    for _, r in PS.iterrows():
        const = not np.isfinite(r.spearman_rowmissing)
        rows.append({"Leverage measure": LAB[r.measure],
                     "$r_S$ with miss.\\ rate": "const." if const else f"{r.spearman_rowmissing:.2f}",
                     "Partial $r_S$ given ref.": "const." if const or not np.isfinite(r.partial_spearman_given_ref)
                     and r.measure.startswith("reference") is False else ("--" if r.measure.startswith("reference") else f"{r.partial_spearman_given_ref:.2f}"),
                     "$r_S$ with ref.": "const." if const else f"{r.spearman_with_ref:.2f}",
                     "Mean (low / mid / high miss.)": f"{r.mean_low:.3f} / {r.mean_mid:.3f} / {r.mean_high:.3f}"})
    PT = pd.DataFrame(rows)
    PT.to_csv(os.path.join(TAB, "rd_pattern.csv"), index=False)
    ter_edges = np.quantile(PL.row_missing, [1 / 3, 2 / 3])
    tex_table(PT, "Pattern-distortion diagnostic on the proteomic predictors (all 375 lines, $p=500$; no response used). "
              "Spearman correlation of each leverage measure with the row missing rate, partial Spearman correlation given the "
              "pattern-free reference (squared norm over the proteins observed in every line), and means by tertile of the "
              "row missing rate.", "tab:rd-pattern", fname="rd_pattern.tex", colfmt="lrrrr",
              note=f"const.: the weight equals 1 for every line. Tertile boundaries of the row missing rate: {ter_edges[0]:.3f} and {ter_edges[1]:.3f}. "
                   f"Reference: {int(PL.shape[0])} lines, proteins observed in every line.")

    # Figure: (a, b) log ratio of each leverage measure to the pattern-free reference against the row
    # missing rate; (c) the resulting leverage weights.  Predictor-only diagnostic (no response used).
    fig = plt.figure(figsize=(7.4, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1], sharey=ax0); ax2 = fig.add_subplot(gs[2])

    def panel(ax, v, c, title):
        ax.scatter(PL.row_missing, v, s=9, color=c, alpha=0.45, lw=0)
        bm = pd.DataFrame({"rm": PL.row_missing.values, "v": np.asarray(v)}).groupby(PL.batch.values).mean()
        ax.scatter(bm.rm, bm.v, s=22, facecolor="white", edgecolor=INK, lw=0.8, zorder=3, label="batch mean")
        k = np.polyfit(PL.row_missing, v, 1)
        xs = np.linspace(PL.row_missing.min(), PL.row_missing.max(), 10)
        ax.plot(xs, np.polyval(k, xs), color=INK, lw=1.3, zorder=4, label=f"LS slope {k[0]:+.2f}")
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("row missing rate", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower left")

    panel(ax0, np.log(PL["zf_norm"] / PL["ref"]), C2, "(a) zero-filled $\\|z_{\\mathrm{obs}}\\|^2/p$")
    panel(ax1, np.log(PL["pace_d2"] / PL["ref"]), C1, "(b) pattern-adjusted $d^2$ (PACE)")
    for ax in (ax0, ax1):
        ax.axhline(0, color=INK2, lw=0.8, ls="--")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax0.set_ylabel("log(measure / reference)", fontsize=8)
    panel(ax2, PL["radial_w"] / np.median(PL["radial_w"]), C2, "(c) leverage weights")
    ax2.plot([PL.row_missing.min(), PL.row_missing.max()], [1.0, 1.0], color=C1, lw=2.0, zorder=5,
             label="PACE $w_x$ (= 1 for all lines)")
    ax2.set_ylabel("weight / median radial weight", fontsize=8)
    h, l = ax2.get_legend_handles_labels()
    ax2.legend(h, [x.replace("batch mean", "radial, batch mean").replace("LS slope", "radial, LS slope") for x in l],
               fontsize=6.0, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.88)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"FigRD1_pattern_distortion.{ext}"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure: relative MAE per drug
fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4), sharex=False)
meths = [m for m in ORDER if m != "Null"]
for ax, dr in zip(axes.ravel(), drugs):
    sub = R[(R.drug == dr) & (R.Method != "Null")]
    g = sub.groupby("Method").relMAE
    mu = g.mean().reindex(meths); se = (g.std(ddof=1) / np.sqrt(g.count())).reindex(meths)
    ypos = np.arange(len(meths))[::-1]
    for yv, m in zip(ypos, meths):
        c = C1 if m == "PACE-CF5" else (C2 if m == "RLPD-MCP" else MUTED)
        ax.errorbar(mu[m], yv, xerr=1.96 * se[m], fmt="o", color=c, ms=4.5, lw=1.2, capsize=0)
    ax.axvline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(ypos); ax.set_yticklabels(meths if ax in axes[:, 0] else [], fontsize=7)
    ax.set_title(DRUGLAB[dr], fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=7)
for ax in axes[1]:
    ax.set_xlabel("held-out MAE / Null MAE")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"FigRD2_relative_mae.{ext}"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- inline values for the manuscript
vals = {}
pcf = R[R.Method == "PACE-CF5"].groupby("drug").relMAE.mean()
vals["rd_pace_relmae_mean"] = f"{pcf.mean():.3f}"
vals["rd_rank_pace"] = f"{rank_mae['PACE-CF5']:.1f}"
vals["rd_best_rank_method"] = rank_mae.idxmin()
vals["rd_best_rank"] = f"{rank_mae.min():.1f}"
with open(os.path.join(TAB, "rd_values.json"), "w") as f:
    json.dump(vals, f, indent=1)
print(json.dumps(vals, indent=1))
print("tables:", sorted(os.listdir(TAB)))

# ---------------------------------------------------------------- candidate datasets (selection stage)
fsel = os.path.join(ROOT, "results", "selection_profile.json")
if os.path.exists(fsel):
    J = json.load(open(fsel))
    a, b, c = J["ccle_proteomics_gdsc_auc"], J["brca_cptac_2020"], J["luad_cptac_2020"]
    rr = lambda d: f"{100 * d['row_missing_quantiles']['0']:.1f}--{100 * d['row_missing_quantiles']['1']:.1f}"
    rows = [
        ["CCLE proteomics + GDSC AUC (selected)", f"{a['n']} lines; {int(a['auc_n_per_drug_quantiles'][0])}--{int(a['auc_n_per_drug_quantiles'][-1])} per drug",
         f"{a['p']:,}", f"{100 * a['cell_missing']:.1f}", rr(a), "drug AUC", "selected"],
        ["CPTAC breast \\citep{krug2020proteogenomic}", str(b["n"]), f"{b['p']:,}", f"{100 * b['cell_missing']:.1f}", rr(b),
         "derived scores only", "small $n$; circular or weak responses"],
        ["CPTAC lung \\citep{gillette2020proteogenomic}", str(c["n"]), f"{c['p']:,}", f"{100 * c['cell_missing']:.1f}", rr(c),
         "age, BMI, pack-years", "small $n$; no clear proteomic response"],
        ["Riboflavin \\citep{buhlmann2014high}", "71", "4,088", "0", "0", "production rate", "no missing values"],
        ["NCI-60 (CellMiner)", "60", "large", "--", "--", "drug activity", "$n$ too small"],
        ["SECOM (UCI)", "1,567", "590", "$\\approx4.5$", "--", "binary", "binary response; $p<n$"],
        ["Communities and Crime (UCI)", "1,994", "122", "--", "--", "crime rate", "$p\\ll n$; column-block missingness"],
        ["DNA-methylation age (GEO)", "$10^2$--$10^3$", "$>4\\times10^5$", "low", "--", "age", "very sparse missingness; not downloadable here"],
    ]
    CT = pd.DataFrame(rows, columns=["Candidate", "$n$", "$p$", "Missing (\\%)", "Row missing (\\%)", "Response", "Decision"])
    CT.to_csv(os.path.join(TAB, "rd_candidates.csv"), index=False)
    tex_table(CT, "Candidate datasets considered before any model was fitted. Profiled quantities for the first three rows are computed by "
              "\\texttt{real\\_data/scripts/01\\_profile\\_candidates.py}; the others are documented properties of datasets that could "
              "not be downloaded in the analysis environment or that failed a requirement outright.", "tab:rd-candidates",
              fname="rd_candidates.tex", colfmt="lrrrrll")


# ---------------------------------------------------------------- supplementary figures
# S1: natural missingness of the analysed p = 500 matrix (predictors only)
Xp = pd.read_csv(os.path.join(ROOT, "data_proc", "X500.csv"), index_col=0)
Lp = pd.read_csv(os.path.join(ROOT, "data_proc", "lines.csv"), index_col=0)
Mm = Xp.isna()
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
rmb = Mm.mean(axis=1).groupby(Lp.loc[Xp.index, "PROTEOMICS_10PLEX_ID"].fillna("unknown").astype(str)).mean().sort_values()
axes[0].bar(np.arange(len(rmb)), rmb.values, color=C1, width=0.8)
axes[0].set_xlabel("TMT batch (sorted)", fontsize=8); axes[0].set_ylabel("row missing rate", fontsize=8)
axes[0].set_title("(a) missing rate by batch", fontsize=8.5); axes[0].set_xticks([])
cm = Mm.mean(axis=0)
axes[1].hist(cm[cm > 0], bins=25, color=C1, edgecolor="white", lw=0.5)
axes[1].set_xlabel("column missing rate", fontsize=8); axes[1].set_ylabel("proteins", fontsize=8)
axes[1].set_title(f"(b) proteins with missing values ({int((cm > 0).sum())} of {len(cm)})", fontsize=8.5)
order = np.argsort(Lp.loc[Xp.index, "PROTEOMICS_10PLEX_ID"].fillna("zz").astype(str).values, kind="stable")
corder = np.argsort(-cm.values, kind="stable")[: int((cm > 0).sum())]
axes[2].imshow(Mm.values[order][:, corder], aspect="auto", cmap=matplotlib.colors.ListedColormap(["white", INK2]),
               interpolation="nearest")
axes[2].set_xlabel("proteins with missing values", fontsize=8); axes[2].set_ylabel("lines (by batch)", fontsize=8)
axes[2].set_title("(c) missingness pattern", fontsize=8.5); axes[2].set_xticks([]); axes[2].set_yticks([])
for ax in axes[:2]:
    ax.tick_params(labelsize=7)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"FigRD_S1_missingness.{ext}"), dpi=200)
plt.close(fig)

# S2: PACE-CF5 final weights against the row missing rate (full-data fits)
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4), sharex=True, sharey=True)
for ax, dr in zip(axes.ravel(), drugs):
    W = pd.read_csv(os.path.join(RAW, f"diag_weights_{dr}.csv"))
    ax.scatter(W.row_missing, W.omega, s=8, color=C1, alpha=0.6, lw=0)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_title(f"{DRUGLAB[dr]} ($n={len(W)}$)", fontsize=8.5)
    ax.tick_params(labelsize=7)
for ax in axes[1]:
    ax.set_xlabel("row missing rate", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("final weight $\\omega_i$", fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"FigRD_S2_weights.{ext}"), dpi=200)
plt.close(fig)
print("figures:", sorted(os.listdir(FIG)))

# ---------------------------------------------------------------- drug panel (selection rule; response values only)
Araw = pd.read_csv(os.path.join(ROOT, "data_proc", "auc_panel.csv"), index_col=0)
Draw = pd.read_csv(os.path.join(ROOT, "data_proc", "drug_annotation.csv"), index_col=0)
rows = []
for dr in PANEL:
    v = Araw[dr].dropna()
    tgt = ""
    if Draw is not None and dr in Draw.index:
        d = str(Draw.loc[dr, "DESCRIPTION"])
        tgt = d.split(";")[0].replace("Target:", "").strip()
    rows.append({"Drug": DRUGLAB[dr], "Target / class (GDSC annotation)": tgt.replace("_", " ").replace("&", "\\&"),
                 "$n$": len(v), "MAD(AUC)": f"{1.4826 * np.median(np.abs(v - v.median())):.3f}",
                 "Median AUC": f"{v.median():.3f}", "AUC $\\le 0.05$": int((v <= 0.05).sum()),
                 "AUC $\\ge 0.99$": int((v >= 0.99).sum())})
DP = pd.DataFrame(rows)
DP.to_csv(os.path.join(TAB, "rd_panel.csv"), index=False)
tex_table(DP, "Pre-specified drug panel: the six drug entries with the largest robust spread $1.4826\\times$MAD of AUC "
          "among entries measured on at least 250 lines with proteomics (one entry per compound). The last two columns count "
          "lines at the lower and upper limits of AUC.", "tab:rd-panel", fname="rd_panel.tex", colfmt="llrrrrr")
print(DP.to_string())
