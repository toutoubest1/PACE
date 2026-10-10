"""Generate every LaTeX table of main.tex / supplement.tex from the raw results of the frozen
reproducibility package (PACE_reproducibility_package.zip -> pace_paper/results/raw).
Usage: python tools/make_tex_tables.py /path/to/pace_paper
All numbers in the manuscript tables are produced by this script; nothing is typed by hand."""
import os, sys, glob
import numpy as np, pandas as pd

PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RAW = os.path.join(PKG, "results", "raw")
OUT = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "tables"))
os.makedirs(OUT, exist_ok=True)

NAME = {"PACE-CF5": "PACE-CF5", "PACE": "PACE-Full", "PACE-SC": "PACE-Full-SP", "PACE-SC-CF5": "PACE-CF5-SP",
        "OLD-RLPD-MCP": "RLPD-MCP", "OLD-RLPD-Lasso": "RLPD-Lasso", "OLD-LPD-Lasso": "LPD-Lasso",
        "Naive-Lasso": "Naive-Lasso", "MICE-Lasso": "MICE-Lasso", "Huber-Lasso": "Huber-Lasso",
        "SparseLTS": "Sparse LTS", "Filt-MICE-SLTS": "Filter-MICE-LTS", "IPW-LPD-Lasso": "IPW-LPD-Lasso",
        "cand-R": "LS repair, no cell flags", "cand-W": "LS repair, winsorised cells ($|z|\\le2.5$)",
        "cand-C": "LS repair (cell flags kept)", "cand-C-Lasso": "LS repair, Lasso penalty",
        "PACE-noY": "no residual weight", "PACE-zerofill-dist": "zero-filled distance",
        "PACE-oldweight-noY": "radial weight $\\min(1,c/\\|x_{\\mathrm{obs}}\\|)$, no residual weight, no flags",
        "PACE-clip-everywhere": "clipping also inside pilot", "PACE-Lasso": "Lasso instead of MCP"}
REG = {"none": "Clean", "xonly": "$x$-shift", "xlev": "Leverage", "yonly": "$y$-only", "xy": "Joint $x/y$",
       "t2err": "$t_2$ errors", "cell6": "Cell $\\pm6$", "cell4": "Cell $\\pm4$",
       "xy_m10": "10\\% miss.", "xy_m30": "30\\% miss.", "xy_m40": "40\\% miss.",
       "xy_hetcol": "Het.\\ columns", "xy_hetrow": "Het.\\ rows", "xy_mar": "MAR", "xy_mnar": "MNAR"}
CONTAM = ["none", "xonly", "xlev", "yonly", "xy", "t2err", "cell6", "cell4"]
MISS = ["xy_m10", "xy", "xy_m30", "xy_m40", "xy_hetcol", "xy_hetrow", "xy_mar", "xy_mnar"]
MAIN_M = ["PACE-CF5", "PACE", "OLD-RLPD-MCP", "OLD-RLPD-Lasso", "OLD-LPD-Lasso", "IPW-LPD-Lasso", "Huber-Lasso",
          "MICE-Lasso", "Naive-Lasso", "SparseLTS", "Filt-MICE-SLTS"]
LOWER = {"L2", "PE", "FDR", "Time"}


def load(d):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, d, "*.csv"))], ignore_index=True)


fm, fp = load("final_main"), load("final_pgrid")


def cell_tab(df, methods, regs, metric, se=True, digits=2, bold=True):
    g = df[df.Method.isin(methods)].groupby(["Method", "regime"])[metric]
    m = g.mean().unstack(); s = (g.std() / np.sqrt(g.count())).unstack()
    rows = []
    for meth in methods:
        cells = []
        for r in regs:
            v = m.loc[meth, r]
            col = m.loc[methods, r]
            best = col.min() if metric in LOWER else col.max()
            txt = f"{v:.{digits}f}"
            if bold and np.isclose(v, best):
                txt = "\\textbf{" + txt + "}"
            if se:
                txt += f"\\,{{\\scriptsize({s.loc[meth, r]:.{digits}f})}}"
            cells.append(txt)
        rows.append(NAME[meth] + " & " + " & ".join(cells) + " \\\\")
    return rows


def write(name, s):
    # shrink any tabular that is wider than the text block; longtables get a smaller font
    s = s.replace("\\begin{tabular}", "\\fitbox{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    if "\\begin{longtable}" in s:
        s = "{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n" + s + "\n}"
    open(os.path.join(OUT, name), "w").write(s)


def main_table(regs, label, caption, fname):
    head = " & ".join(REG[r] for r in regs)
    s = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", "\\scriptsize",
         "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(regs) + "}", "\\toprule",
         "Method & " + head + " \\\\", "\\midrule",
         f"\\multicolumn{{{len(regs)+1}}}{{l}}{{\\emph{{(a) Coefficient error $\\|\\hat\\beta-\\beta^*\\|_2$, mean (s.e.)}}}}\\\\"]
    s += cell_tab(fm, MAIN_M, regs, "L2")
    s += ["\\midrule", f"\\multicolumn{{{len(regs)+1}}}{{l}}{{\\emph{{(b) F1 score of the selected support}}}}\\\\"]
    s += cell_tab(fm, MAIN_M, regs, "F1", se=False)
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(fname, "\n".join(s) + "\n")


main_table(CONTAM, "tab:contam",
           "Held-out simulation, contamination regimes ($n=200$, $p=300$, MCAR rate $0.2$, 20 replicates, seeds $7000$--$7019$). "
           "Bold: best mean in the column. RLPD and LPD denote the radial-weighting baselines of Section~\\ref{sec-sim-design} "
           "(corrected implementation).", "tab_contam.tex")
main_table(MISS, "tab:miss",
           "Held-out simulation, missingness severity, heterogeneity and mechanism under joint $x/y$ contamination "
           "($n=200$, $p=300$, 20 replicates). Columns 1--4: MCAR rate $0.1,0.2,0.3,0.4$; Het.\\ columns: column rates "
           "$\\mathrm{U}[0.04,0.36]$; Het.\\ rows: half the rows $5\\%$, half $35\\%$ missing (MCAR); MAR and MNAR as in "
           "Section~\\ref{sec-sim-design}.", "tab_miss.tex")

# ---------------- ablation (main: compact)
ABL = ["PACE", "PACE-noY", "PACE-zerofill-dist", "PACE-oldweight-noY", "cand-C", "cand-R", "cand-W", "PACE-Lasso"]
ABL_REG = ["none", "xlev", "yonly", "xy", "cell6", "xy_hetrow", "xy_mnar"]
s = ["\\begin{table}[!htbp]", "\\centering",
     "\\caption{Ablation of the full-sample construction (held-out, $p=300$, 20 replicates): mean coefficient error. "
     "Each row removes or replaces components of PACE-Full; ``LS repair'' replaces eigenvalue clipping by linear-shrinkage repair at the final fit, so the three LS rows isolate the cell-handling step relative to each other.}\\label{tab:ablation}",
     "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(ABL_REG) + "}", "\\toprule",
     "Variant & " + " & ".join(REG[r] for r in ABL_REG) + " \\\\", "\\midrule"]
g = fm[fm.Method.isin(ABL)].groupby(["Method", "regime"]).L2.mean().unstack()
for m in ABL:
    s.append(("PACE-Full (reference)" if m == "PACE" else NAME[m]) + " & " +
             " & ".join(f"{g.loc[m, r]:.2f}" for r in ABL_REG) + " \\\\")
s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_ablation.tex", "\n".join(s) + "\n")

# ---------------- cross-fitting validation (main)
dc = load("dev_crossfit")
DEVR = ["none", "xonly", "yonly", "xy", "t2err", "cell6"]
gd = dc.groupby(["Method", "regime"]).L2.mean().unstack()
allr = pd.concat([fm, fp], ignore_index=True)
gg = allr.groupby(["regime", "p", "Method"])[["L2", "F1", "Time"]].mean().unstack()
ratio = gg[("L2", "PACE-CF5")] / gg[("L2", "PACE")]
dF1 = gg[("F1", "PACE-CF5")] - gg[("F1", "PACE")]
s = ["\\begin{table}[!htbp]", "\\centering",
     "\\caption{Theory-aligned versions of PACE. Panel (a): development seeds $2000$--$2009$ ($p=300$, 10 replicates), "
     "mean coefficient error. Panel (b): pre-specified comparison of PACE-CF5 with PACE-Full on all 27 held-out "
     "regime--dimension cells.}\\label{tab:crossfit}",
     "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{l" + "c" * len(DEVR) + "}", "\\toprule",
     "\\multicolumn{%d}{l}{\\emph{(a) Development seeds}}\\\\" % (len(DEVR) + 1),
     "Version & " + " & ".join(REG[r] for r in DEVR) + " \\\\", "\\midrule"]
for m, lab in [("PACE", "PACE-Full"), ("PACE-split", "single split (fit on $n/2$)"),
               ("PACE-halfsample", "PACE-Full on a half sample"), ("PACE-crossfit", "2-fold cross-fitting"),
               ("PACE-crossfit5", "5-fold cross-fitting (PACE-CF5)")]:
    s.append(lab + " & " + " & ".join(f"{gd.loc[m, r]:.2f}" for r in DEVR) + " \\\\")
s += ["\\midrule", "\\multicolumn{%d}{l}{\\emph{(b) Held-out seeds, PACE-CF5 versus PACE-Full (27 cells)}}\\\\" % (len(DEVR) + 1),
      f"\\multicolumn{{{len(DEVR)+1}}}{{l}}{{Error ratio: mean {ratio.mean():.3f}, median {ratio.median():.3f}, range "
      f"[{ratio.min():.3f}, {ratio.max():.3f}]; pre-specified thresholds: mean $\\le1.05$, every cell $\\le1.15$.}}\\\\",
      f"\\multicolumn{{{len(DEVR)+1}}}{{l}}{{F1 difference: mean {dF1.mean():+.3f}, range [{dF1.min():+.3f}, {dF1.max():+.3f}].}}\\\\"]
t300 = fm[fm.Method.isin(["PACE", "PACE-CF5"])].groupby("Method").Time.mean()
t1000 = fp[(fp.p == 1000) & fp.Method.isin(["PACE", "PACE-CF5"])].groupby("Method").Time.mean()
s += [f"\\multicolumn{{{len(DEVR)+1}}}{{l}}{{Mean single-core time (s): $p=300$: {t300['PACE']:.1f} vs {t300['PACE-CF5']:.1f}; "
      f"$p=1000$: {t1000['PACE']:.1f} vs {t1000['PACE-CF5']:.1f} (PACE-Full vs PACE-CF5).}}\\\\",
      "\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_crossfit.tex", "\n".join(s) + "\n")

# ---------------- masking table (main)
det = pd.read_csv(os.path.join(RAW, "studies", "pattern_detection.csv"))
s = ["\\begin{table}[!htbp]", "\\centering",
     "\\caption{Outlier masking by missingness ($n=400$, $p=300$, 20 replicates, clean data with $10\\%$ of rows shifted by "
     "$+1.5$ on half the coordinates; row missing rates assigned at random, MCAR). Entries are mean leverage weights. "
     "``Radial'' is $\\min(1,c/\\|x_{i,\\mathrm{obs}}\\|)$; ``zero-filled'' applies the PACE weight function to the zero-filled "
     "standardised norm; ``PACE'' uses the pattern-adjusted distance.}\\label{tab:masking}",
     "\\small", "\\begin{tabular}{ccccccc}", "\\toprule",
     " & \\multicolumn{2}{c}{Radial} & \\multicolumn{2}{c}{Zero-filled} & \\multicolumn{2}{c}{PACE} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
     "Row missing rate & clean & outlier & clean & outlier & clean & outlier \\\\", "\\midrule"]
for mr in sorted(det.miss.unique()):
    a = det[(det.miss == mr) & (~det.out)].iloc[0]; b = det[(det.miss == mr) & (det.out)].iloc[0]
    s.append(f"{mr:.1f} & {a.w_old:.3f} & {b.w_old:.3f} & {a.w_zf:.3f} & {b.w_zf:.3f} & {a.w_new:.3f} & {b.w_new:.3f} \\\\")
s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_masking.tex", "\n".join(s) + "\n")

# ================= SUPPLEMENT TABLES =================
MET = ["L2", "PE", "AUC", "F1", "TPR", "FDR", "Size", "Time"]
SUPP_M = MAIN_M + ["PACE-SC", "PACE-SC-CF5"]
parts = []
for r in CONTAM + [x for x in MISS if x != "xy"]:
    d = fm[(fm.regime == r) & fm.Method.isin(SUPP_M)]
    g = d.groupby("Method")[MET].agg(["mean", "std", "count"])
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{Held-out results, regime: {REG[r]} ($n=200$, $p=300$, "
             f"20 replicates). Mean (s.e.) for L2 and PE; means otherwise; Time in seconds (single core).}}\\label{{stab:full-{r}}}",
             "\\scriptsize", "\\begin{tabular}{lcccccccc}", "\\toprule", "Method & L2 & PE & AUC & F1 & TPR & FDR & Size & Time \\\\", "\\midrule"]
    for m in SUPP_M:
        row = [NAME[m]]
        for k in MET:
            mu = g.loc[m, (k, "mean")]; se = g.loc[m, (k, "std")] / np.sqrt(g.loc[m, (k, "count")])
            if k in ("L2", "PE"):
                row.append(f"{mu:.2f} ({se:.2f})")
            elif k == "Size":
                row.append(f"{mu:.1f}")
            elif k == "Time":
                row.append(f"{mu:.1f}")
            else:
                row.append(f"{mu:.2f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    parts.append("\n".join(lines))
write("supp_full_regimes.tex", "\n\n".join(parts) + "\n")

# dimension tables
PG_M = ["PACE-CF5", "PACE", "PACE-SC", "PACE-SC-CF5", "OLD-RLPD-MCP", "OLD-RLPD-Lasso", "Huber-Lasso", "MICE-Lasso", "Filt-MICE-SLTS"]
both = pd.concat([fp, fm[fm.regime.isin(["none", "yonly", "xy", "cell6"])]], ignore_index=True)
parts = []
for metric, lab in [("L2", "coefficient error"), ("F1", "F1"), ("FDR", "false discovery rate"), ("TPR", "true positive rate"), ("Time", "time (s)")]:
    lines = ["\\begin{table}[htbp]", "\\centering",
             f"\\caption{{Dimension study, {lab} (held-out; $p/n=0.5,1.5,2.5,5$; 20, 20, 10, 10 replicates).}}\\label{{stab:pgrid-{metric}}}",
             "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "cccc" * 4 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{REG[r]}}}" for r in ["none", "yonly", "xy", "cell6"]) + " \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}",
             "Method & " + " & ".join(["100", "300", "500", "1000"] * 4) + " \\\\", "\\midrule"]
    g = both.groupby(["Method", "regime", "p"])[metric].mean()
    for m in PG_M:
        vals = []
        for r in ["none", "yonly", "xy", "cell6"]:
            for p in [100, 300, 500, 1000]:
                v = g.get((m, r, p), np.nan)
                vals.append("--" if np.isnan(v) else (f"{v:.1f}" if metric == "Time" else f"{v:.2f}"))
        lines.append(NAME[m] + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    parts.append("\n".join(lines))
write("supp_pgrid.tex", "\n\n".join(parts) + "\n")

# full ablations
ABL2 = ["PACE", "PACE-noY", "PACE-zerofill-dist", "PACE-oldweight-noY", "cand-R", "cand-W", "cand-C", "cand-C-Lasso",
        "PACE-clip-everywhere", "PACE-Lasso", "PACE-CF5"]
parts = []
for metric in ["L2", "F1"]:
    for regs, tag in [(CONTAM, "a"), ([x for x in MISS], "b")]:
        g = fm[fm.Method.isin(ABL2)].groupby(["Method", "regime"])[metric].mean().unstack()
        lines = ["\\begin{table}[htbp]", "\\centering",
                 f"\\caption{{Ablations, {('coefficient error' if metric=='L2' else 'F1')} (held-out, $p=300$, 20 replicates).}}\\label{{stab:abl-{metric}-{tag}}}",
                 "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(regs) + "}", "\\toprule",
                 "Variant & " + " & ".join(REG[r] for r in regs) + " \\\\", "\\midrule"]
        for m in ABL2:
            lab = "PACE-Full (reference)" if m == "PACE" else NAME[m]
            lines.append(lab + " & " + " & ".join(f"{g.loc[m, r]:.2f}" for r in regs) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        parts.append("\n".join(lines))
write("supp_ablation.tex", "\n\n".join(parts) + "\n")

# stabiliser
st = pd.read_csv(os.path.join(RAW, "studies", "stabiliser_raw.csv"))
g = st.groupby(["regime", "stab"])[["L2", "PE", "AUC", "F1", "FDR", "Size"]].mean()
SL = {"clip_final": "eigenvalue clipping at final fit (PACE)", "lpd": "linear shrinkage (feasibility boundary)",
      "cvalpha": "linear shrinkage, intensity by moment-CV risk"}
lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Positive-semidefinite repair of the final moment matrix (full-sample PACE, held-out seeds $7000$--$7009$, $p=300$).}\\label{stab:stabiliser}",
         "\\scriptsize", "\\begin{tabular}{llcccccc}", "\\toprule", "Regime & Repair & L2 & PE & AUC & F1 & FDR & Size \\\\", "\\midrule"]
for r in ["none", "yonly", "xy", "cell6"]:
    for k in ["clip_final", "lpd", "cvalpha"]:
        v = g.loc[(r, k)]
        lines.append(f"{REG[r]} & {SL[k]} & {v.L2:.2f} & {v.PE:.2f} & {v.AUC:.3f} & {v.F1:.2f} & {v.FDR:.2f} & {v.Size:.1f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("supp_stabiliser.tex", "\n".join(lines) + "\n")

# pilot sensitivity (dev + held-out rule)
dp = load("dev_pilot")
PV = [("PACE", "concentration-step pilot (frozen)"), ("PACE-pilot-screen", "screened LTS only"),
      ("PACE-pilot-screen+cstep", "starts \\{all, zero, screened\\}"), ("PACE-pilot-screen_cstep", "screened start + C-steps (SP)"),
      ("PACE-pilot-screen_cstep+cstep", "\\{all, zero, screened+C-steps\\}"), ("PACE-pilot-screen_cstep-K38", "SP with $K=38$")]
cols = [(300, "none"), (300, "xonly"), (300, "t2err"), (300, "xy"), (300, "cell6"), (300, "yonly"), (500, "yonly"), (1000, "yonly")]
g = dp.groupby(["Method", "p", "regime"]).L2.mean()
lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Development study of high-dimensional pilots (development seeds $2000$--$2009$, 10 replicates; mean coefficient error of the full-sample estimator). Columns: regime and $p$.}\\label{stab:pilot-dev}",
         "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "Pilot & " + " & ".join(f"{REG[r]}, {p}" for p, r in cols) + " \\\\", "\\midrule"]
for m, lab in PV:
    lines.append(lab + " & " + " & ".join(("--" if np.isnan(g.get((m, p, r), np.nan)) else f"{g.get((m, p, r)):.2f}") for p, r in cols) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("supp_pilot_dev.tex", "\n".join(lines) + "\n")

dec = pd.DataFrame({"PACE": gg[("L2", "PACE")], "SP": gg[("L2", "PACE-SC")],
                    "r": gg[("L2", "PACE-SC")] / gg[("L2", "PACE")], "dF1": gg[("F1", "PACE-SC")] - gg[("F1", "PACE")],
                    "CF5": gg[("L2", "PACE-CF5")], "rcf": ratio, "dcf": dF1})
lines = ["\\begin{longtable}{lcccccccc}",
         "\\caption{Pre-specified decision quantities on the held-out seeds. SP: screened pilot. Ratios are mean-error ratios relative to PACE-Full; differences are F1 differences.}\\label{stab:decision}\\\\",
         "\\toprule", "Regime & $p$ & PACE-Full & PACE-Full-SP & ratio & $\\Delta$F1 & PACE-CF5 & ratio & $\\Delta$F1 \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "Regime & $p$ & PACE-Full & PACE-Full-SP & ratio & $\\Delta$F1 & PACE-CF5 & ratio & $\\Delta$F1 \\\\", "\\midrule", "\\endhead",
         "\\bottomrule", "\\endlastfoot"]
for (r, p), v in dec.iterrows():
    lines.append(f"{REG[r]} & {p} & {v.PACE:.3f} & {v.SP:.3f} & {v.r:.3f} & {v.dF1:+.3f} & {v.CF5:.3f} & {v.rcf:.3f} & {v.dcf:+.3f} \\\\")
lines += ["\\end{longtable}"]
write("supp_decision.tex", "\n".join(lines) + "\n")

# cross-fitting development (full)
MM = [("PACE", "PACE-Full"), ("PACE-split", "single split"), ("PACE-split-ls", "single split, local scale"),
      ("PACE-halfsample", "PACE-Full on half sample"), ("PACE-crossfit", "2-fold"), ("PACE-crossfit-ls", "2-fold, local scale"),
      ("PACE-crossfit5", "5-fold (PACE-CF5)"), ("PACE-crossfit5-ls", "5-fold, local scale")]
parts = []
for metric in ["L2", "F1", "Time"]:
    g = dc.groupby(["Method", "regime"])[metric].mean().unstack()
    lines = ["\\begin{table}[htbp]", "\\centering",
             f"\\caption{{Development study of sample splitting and cross-fitting ($p=300$, seeds $2000$--$2009$): {metric}.}}\\label{{stab:cf-dev-{metric}}}",
             "\\scriptsize", "\\begin{tabular}{l" + "c" * len(DEVR) + "}", "\\toprule",
             "Version & " + " & ".join(REG[r] for r in DEVR) + " \\\\", "\\midrule"]
    for m, lab in MM:
        lines.append(lab + " & " + " & ".join(f"{g.loc[m, r]:.2f}" for r in DEVR) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    parts.append("\n".join(lines))
write("supp_cf_dev.tex", "\n\n".join(parts) + "\n")

# rate check
rc = pd.read_csv(os.path.join(RAW, "studies", "rate_check.csv"))
lines = ["\\begin{longtable}{cccccc}",
         "\\caption{Numerical check of Theorem~\\ref{M-thm:dev}: maximum entrywise error of unweighted pairwise-Hajek moments ($p=200$, AR(0.5), flag threshold $3.5$, coordinated casewise outliers placed at $\\pm3$, just inside the threshold; mean over 5 replicates).}\\label{stab:rate}\\\\",
         "\\toprule", "$n$ & $\\pi$ & $\\varepsilon$ & max-norm error & $\\sqrt{\\log p/n}/\\pi$ & \\\\", "\\midrule", "\\endfirsthead",
         "\\bottomrule", "\\endlastfoot"]
for _, v in rc.iterrows():
    lines.append(f"{int(v.n)} & {v.pi:.1f} & {v.eps:.2f} & {v.maxerr:.3f} & {v['sqrt(log p/n)/pi']:.3f} & \\\\")
lines += ["\\end{longtable}"]
write("supp_rate.tex", "\n".join(lines) + "\n")
d0 = rc[rc.eps == 0]
slope = np.polyfit(np.log(d0["sqrt(log p/n)/pi"]), np.log(d0.maxerr), 1)[0]
write("val_rate_slope.tex", f"{slope:.2f}")

# kappa + pattern weights + pairs
kr = pd.read_csv(os.path.join(RAW, "studies", "kappa_raw.csv"))
k = kr.groupby("method")[["kappa", "rel_err"]].agg(["mean", "std"])
lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Population-level coefficient bias of weighted moment equations under row-heterogeneous MCAR ($n=20{,}000$, $p=60$, half the rows $5\\%$ and half $50\\%$ missing, unpenalised solutions, 5 replicates). $\\kappa=\\langle\\hat\\beta_S,\\beta^*_S\\rangle/\\|\\beta^*_S\\|^2$.}\\label{stab:kappa}",
         "\\small", "\\begin{tabular}{lcc}", "\\toprule", "Construction & $\\kappa$ mean (sd) & relative error mean (sd) \\\\", "\\midrule"]
KL = {"old weight + column IPW (old RLPD)": "radial weight, column-rate IPW (RLPD moments)",
      "old weight + pairwise Hajek": "radial weight, pairwise-Hajek moments",
      "Hajek-distance weight + pairwise Hajek": "pattern-adjusted radial weight, pairwise-Hajek",
      "unweighted pairwise Hajek": "unweighted pairwise-Hajek"}
for m in KL:
    lines.append(f"{KL[m]} & {k.loc[m, ('kappa','mean')]:.4f} ({k.loc[m, ('kappa','std')]:.4f}) & {k.loc[m, ('rel_err','mean')]:.4f} ({k.loc[m, ('rel_err','std')]:.4f}) \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("supp_kappa.tex", "\n".join(lines) + "\n")

pw = pd.read_csv(os.path.join(RAW, "studies", "pattern_weights.csv"))
lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Distances and leverage weights of clean rows by row missing rate ($n=400$, $p=300$, 20 replicates). Column-HT is the Horvitz--Thompson distance without row normalisation.}\\label{stab:pattern}",
         "\\small", "\\begin{tabular}{cccccccc}", "\\toprule",
         "miss.\\ rate & latent $\\|z\\|^2/p$ & zero-filled & column-HT & PACE $d^2$ & radial weight & PACE weight \\\\", "\\midrule"]
for _, v in pw.iterrows():
    lines.append(f"{v.miss:.1f} & {v.latent:.3f} & {v.zf_norm2:.3f} & {v.ht:.3f} & {v.hajek:.3f} & {v.w_old:.3f} & {v.w_new:.3f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("supp_pattern.tex", "\n".join(lines) + "\n")
pp = pd.read_csv(os.path.join(RAW, "studies", "pattern_pairs.csv"))
for key, val in {"npairs": f"{len(pp)}", "pairzf": f"{pp.dist_ratio_zf.mean():.3f}", "pairhaj": f"{pp.dist_ratio_hajek.mean():.3f}",
                 "pairwold": f"{pp.wratio_old.mean():.2f}", "pairwoldsd": f"{pp.wratio_old.std():.2f}",
                 "pairwnew": f"{pp.wratio_new.mean():.3f}"}.items():
    write(f"val_{key}.tex", val)

# paired differences
rows = []
for (r, p), d in allr.groupby(["regime", "p"]):
    w = d.pivot_table(index="rep", columns="Method", values="L2")
    rr = [REG[r], str(p)]
    for c in ["OLD-RLPD-MCP", "Huber-Lasso", "MICE-Lasso", "Filt-MICE-SLTS"]:
        x = w["PACE-CF5"] - w[c]
        rr.append(f"{x.mean():+.2f} ({x.std()/np.sqrt(len(x)):.2f}) {int((x<0).sum())}/{len(x)}")
    rows.append(" & ".join(rr) + " \\\\")
lines = ["\\begin{longtable}{llcccc}",
         "\\caption{Paired differences in coefficient error, PACE-CF5 minus competitor (held-out). Mean (s.e.) and number of replicates in which PACE-CF5 is better.}\\label{stab:paired}\\\\",
         "\\toprule", "Regime & $p$ & RLPD-MCP & Huber-Lasso & MICE-Lasso & Filter-MICE-LTS \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "Regime & $p$ & RLPD-MCP & Huber-Lasso & MICE-Lasso & Filter-MICE-LTS \\\\", "\\midrule", "\\endhead",
         "\\bottomrule", "\\endlastfoot"] + rows + ["\\end{longtable}"]
write("supp_paired.tex", "\n".join(lines) + "\n")
print("tables written to", OUT)
