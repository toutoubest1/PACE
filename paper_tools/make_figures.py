"""figures from raw results (PNG + PDF in results/figures).
Palette: validated categorical slots 1-3 (blue, orange, aqua); marker shape is a secondary encoding."""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
ROOT = PKG
sys.path.insert(0, PKG)
from experiments.config import REGIMES

RAW = os.path.join(ROOT, "results", "raw"); OUT = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "figures"))
os.makedirs(OUT, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
SER = [("PACE-CF5", "PACE-CF5", C1, "o"), ("OLD-RLPD-MCP", "RLPD-MCP", C2, "s"),
       ("Huber-Lasso", "Huber-Lasso", C3, "^")]
NAMES = {"none": "no contamination", "xonly": "x-only shift", "xlev": "gross leverage", "yonly": "y-only outliers",
         "xy": "joint x/y", "t2err": "t2 errors", "cell6": "cellwise ±6", "cell4": "cellwise ±4",
         "xy_m10": "x/y, 10% missing", "xy_m30": "x/y, 30% missing", "xy_m40": "x/y, 40% missing",
         "xy_hetcol": "x/y, het. columns", "xy_hetrow": "x/y, het. rows", "xy_mar": "x/y, MAR", "xy_mnar": "x/y, MNAR"}


def load(d):
    fs = glob.glob(os.path.join(RAW, d, "*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_main(fm):
    regs = [r for r in REGIMES if r in fm.regime.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    for ax, met, lab in [(axes[0], "L2", "coefficient error  ‖β̂ − β*‖₂"), (axes[1], "F1", "F1 (support recovery)")]:
        for k, (m, lbl, c, mk) in enumerate(SER):
            d = fm[fm.Method == m].groupby("regime")[met]
            mu = d.mean().reindex(regs); se = (d.std() / np.sqrt(d.count())).reindex(regs)
            y = np.arange(len(regs)) + (k - 1) * 0.22
            ax.errorbar(mu, y, xerr=1.96 * se, fmt=mk, color=c, ms=5, lw=1.2, capsize=0, label=lbl,
                        mec="white", mew=0.6)
        ax.set_xlabel(lab)
        ax.set_yticks(np.arange(len(regs))); ax.set_yticklabels([NAMES[r] for r in regs])
        ax.invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    pass
    save(fig, "Fig1_main_L2_F1")


def fig_pgrid(both):
    regs = ["none", "yonly", "xy", "cell6"]
    ser = [("PACE-CF5", "PACE-CF5", C1, "o"), ("PACE", "PACE-Full", "#4a3aa7", "^"),
           ("OLD-RLPD-MCP", "RLPD-MCP", C2, "s")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, r in zip(axes, regs):
        d = both[both.regime == r]
        for k, (m, lbl, c, mk) in enumerate(ser):
            g = d[d.Method == m].groupby("p").L2
            mu = g.mean(); se = g.std() / np.sqrt(g.count())
            ax.errorbar(mu.index / 200 + (k - 1) * 0.08, mu.values, yerr=1.96 * se.values, marker=mk, color=c, lw=2, ms=5,
                        capsize=0, label=lbl, mec="white", mew=0.6)
        ax.set_title(NAMES[r], fontsize=9, color=INK); ax.set_xlabel("p / n"); ax.set_xticks([0.5, 1.5, 2.5, 5])
    axes[0].set_ylabel("‖β̂ − β*‖₂"); axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "Fig2_dimension_study")


def fig_pattern():
    p = os.path.join(RAW, "studies", "pattern_weights.csv")
    if not os.path.exists(p): return
    w = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax = axes[0]
    ax.plot(w.miss, w.zf_norm2, marker="s", color=C2, lw=2, label="zero-filled ‖z_obs‖²/p")
    ax.plot(w.miss, w.hajek, marker="o", color=C1, lw=2, label="pattern-adjusted d² (PACE)")
    ax.plot(w.miss, w.latent, color=INK2, lw=1, ls="--", label="latent ‖z‖²/p")
    ax.set_xlabel("row missing rate"); ax.set_ylabel("distance (clean rows)"); ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(w.miss, w.w_old / w.w_old.iloc[0], marker="s", color=C2, lw=2, label="radial weight min(1, c/‖x_obs‖)")
    ax.plot(w.miss, w.w_new / w.w_new.iloc[0], marker="o", color=C1, lw=2, label="PACE leverage weight")
    ax.set_xlabel("row missing rate"); ax.set_ylabel("weight relative to complete row"); ax.legend(fontsize=7)
    save(fig, "Fig3_missingness_pattern")


if __name__ == "__main__":
    fm = load("final_main"); fp = load("final_pgrid")
    if fm is not None:
        fig_main(fm)
    if fp is not None:
        both = pd.concat([fp, fm[fm.regime.isin(["none", "yonly", "xy", "cell6"])]]) if fm is not None else fp
        fig_pgrid(both)
    fig_pattern()
    print("figures written to", OUT)
