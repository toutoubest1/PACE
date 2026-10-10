"""Convert raw per-replicate CSVs into the tables (Markdown, CSV and LaTeX).  """
import os, sys, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.config import REGIMES

RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)

LABEL = {
    "PACE": "PACE full-sample", "PACE-SC": "PACE full-sample, screened pilot", "PACE-CF5": "PACE-CF5 (frozen)", "PACE-SC-CF5": "PACE-CF5, screened pilot",
    "OLD-RLPD-MCP": "OLD RLPD-MCP*", "OLD-RLPD-Lasso": "OLD RLPD-Lasso*", "OLD-LPD-Lasso": "OLD LPD-Lasso*",
    "Naive-Lasso": "Naive-Lasso", "MICE-Lasso": "MICE-Lasso", "Huber-Lasso": "Huber-Lasso",
    "SparseLTS": "Sparse LTS", "Filt-MICE-SLTS": "Filter-MICE-SparseLTS", "IPW-LPD-Lasso": "IPW-LPD-Lasso",
    "cand-R": "rowwise only (R)", "cand-W": "winsorised cells (W)", "cand-C": "cell flags + LPD (C)",
    "cand-C-Lasso": "C with Lasso", "PACE-noY": "PACE without residual weight",
    "PACE-zerofill-dist": "PACE with zero-filled distance", "PACE-oldweight-noY": "old weight + new moments/tuning",
    "PACE-clip-everywhere": "clipping in pilot too", "PACE-Lasso": "PACE with Lasso penalty",
}
MAIN = ["PACE-CF5", "PACE", "PACE-SC", "PACE-SC-CF5", "OLD-RLPD-MCP", "OLD-RLPD-Lasso", "OLD-LPD-Lasso", "Naive-Lasso", "MICE-Lasso",
        "Huber-Lasso", "SparseLTS", "Filt-MICE-SLTS", "IPW-LPD-Lasso"]
ABL = ["PACE", "cand-R", "cand-W", "cand-C", "cand-C-Lasso", "PACE-noY", "PACE-zerofill-dist",
       "PACE-oldweight-noY", "PACE-clip-everywhere", "PACE-Lasso", "OLD-RLPD-MCP"]
REG_ORDER = list(REGIMES)
LOWER_BETTER = {"L2", "PE", "FDR", "Time"}


def load(d):
    fs = sorted(glob.glob(os.path.join(RAW, d, "*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def save(name, df, md_text=None):
    df.to_csv(os.path.join(OUT, name + ".csv"))
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write(md_text if md_text is not None else df.to_markdown())
    try:
        df.to_latex(os.path.join(OUT, name + ".tex"), escape=True)
    except Exception:
        pass


def metric_by_regime(df, methods, metric, cols="regime", se=True, digits=2):
    df = df[df.Method.isin(methods)]
    g = df.groupby(["Method", cols])[metric]
    m = g.mean().unstack(); s = (g.std() / np.sqrt(g.count())).unstack()
    order = [c for c in REG_ORDER if c in m.columns] if cols == "regime" else sorted(m.columns)
    m, s = m[order], s[order]
    m = m.loc[[x for x in methods if x in m.index]]; s = s.loc[m.index]
    best = m.min() if metric in LOWER_BETTER else m.max()
    txt = m.copy().astype(object)
    for c in m.columns:
        for i in m.index:
            v = f"{m.loc[i, c]:.{digits}f}" + (f" ({s.loc[i, c]:.{digits}f})" if se else "")
            if np.isclose(m.loc[i, c], best[c]):
                v = f"**{v}**"
            txt.loc[i, c] = v
    txt.index = [LABEL.get(i, i) for i in txt.index]
    m.index = txt.index
    return m, txt


def main():
    fm = load("final_main")
    if fm is not None:
        reps = fm.groupby("regime").rep.nunique()
        note = f"\n\nFinal held-out seeds 7000+r, n=200, p=300; replicates per regime: {reps.min()}-{reps.max()}. Mean (s.e.). * = rejected manuscript's method with the boolean-mask and eigenvalue-floor defects fixed.\n"
        for met in ["L2", "PE", "AUC", "F1", "TPR", "FDR", "Size", "Time"]:
            m, t = metric_by_regime(fm, MAIN, met, se=met in ("L2", "PE", "F1"))
            save(f"T1_main_{met}", m, f"### {met}, main comparison\n\n" + t.to_markdown() + note)
        m, t = metric_by_regime(fm, ABL, "L2")
        save("T2_ablation_L2", m, "### L2, ablations\n\n" + t.to_markdown() + note)
        m, t = metric_by_regime(fm, ABL, "F1")
        save("T2_ablation_F1", m, "### F1, ablations\n\n" + t.to_markdown() + note)
        # per-regime full metric tables
        parts = []
        for reg in REG_ORDER:
            d = fm[(fm.regime == reg) & fm.Method.isin(MAIN)]
            if d.empty: continue
            g = d.groupby("Method")[["L2", "PE", "AUC", "F1", "TPR", "FDR", "Size", "Time"]].mean()
            g = g.loc[[x for x in MAIN if x in g.index]].round(3)
            g.index = [LABEL.get(i, i) for i in g.index]
            parts.append(f"#### {reg}\n\n" + g.to_markdown())
            g.to_csv(os.path.join(OUT, f"T3_full_{reg}.csv"))
        open(os.path.join(OUT, "T3_full_all_regimes.md"), "w").write("\n\n".join(parts) + note)
    fp = load("final_pgrid")
    if fp is not None:
        both = pd.concat([fp, fm[fm.regime.isin(["none", "yonly", "xy", "cell6"])]]) if fm is not None else fp
        meths = ["PACE-CF5", "PACE", "PACE-SC", "PACE-SC-CF5", "OLD-RLPD-MCP", "OLD-RLPD-Lasso", "MICE-Lasso", "Huber-Lasso", "Filt-MICE-SLTS"]
        both = both[both.Method.isin(meths)]
        for met in ["L2", "F1", "FDR", "Time"]:
            t = both.pivot_table(index=["regime", "Method"], columns="p", values=met, aggfunc="mean").round(3)
            t = t.rename(index=LABEL)
            reps = both.groupby("p").rep.nunique().to_dict()
            save(f"T4_pgrid_{met}", t, f"### {met} by dimension (n=200)\n\n" + t.to_markdown() + f"\n\nReplicates per p: {reps}\n")
    # pilot decision and theory-version decision
    if fm is not None:
        allr = pd.concat([fm, fp]) if fp is not None else fm
        rows = []
        for (reg, p), d in allr.groupby(["regime", "p"]):
            g = d.groupby("Method")[["L2", "F1"]].mean()
            if "PACE" not in g.index: continue
            r = dict(regime=reg, p=p, PACE_L2=g.loc["PACE", "L2"], PACE_F1=g.loc["PACE", "F1"])
            for v in ["PACE-SC", "PACE-CF5", "PACE-SC-CF5"]:
                if v in g.index:
                    r[f"{v}_L2"] = g.loc[v, "L2"]; r[f"{v}_F1"] = g.loc[v, "F1"]
                    r[f"{v}_L2ratio"] = g.loc[v, "L2"] / g.loc["PACE", "L2"]
            rows.append(r)
        dec = pd.DataFrame(rows).round(3)
        save("T5_pilot_and_crossfit_decision", dec.set_index(["regime", "p"]))
    # studies
    st = os.path.join(RAW, "studies")
    if os.path.exists(os.path.join(st, "stabiliser_raw.csv")):
        s = pd.read_csv(os.path.join(st, "stabiliser_raw.csv"))
        t = s.groupby(["regime", "stab"])[["L2", "PE", "AUC", "F1", "FDR", "Size"]].mean().round(3)
        save("T6_stabiliser", t)
    for f, nm in [("pattern_weights.csv", "T7a_pattern_weights"), ("pattern_detection.csv", "T7b_outlier_masking"),
                  ("kappa_bias.csv", "T7c_population_bias"), ("rate_check.csv", "T8_theorem1_rate_check")]:
        p = os.path.join(st, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            save(nm, d, d.to_markdown(index=False))
    if os.path.exists(os.path.join(st, "pattern_pairs.csv")):
        pp = pd.read_csv(os.path.join(st, "pattern_pairs.csv"))
        t = pp.drop(columns=[c for c in pp.columns if c.startswith("Unnamed") or c == "rep"]).agg(["mean", "std"]).round(3)
        save("T7d_matched_pairs", t)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
