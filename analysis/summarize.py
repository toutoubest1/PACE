"""Summaries of raw per-replicate CSVs.
python analysis/summarize.py results/raw/<dir> [metric ...]"""
import sys, glob, os
import pandas as pd


def load(d):
    fs = glob.glob(os.path.join(d, "*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


if __name__ == "__main__":
    df = load(sys.argv[1])
    mets = sys.argv[2:] or ["L2", "F1"]
    for m in mets:
        t = df.pivot_table(index="Method", columns=["p", "regime"], values=m, aggfunc="mean").round(3)
        print(f"== {m} (n reps: {df.groupby(['p','regime']).rep.nunique().min()}-{df.groupby(['p','regime']).rep.nunique().max()})")
        print(t.to_string())
