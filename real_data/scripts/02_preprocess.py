"""Deterministic preprocessing ."""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = os.path.join(ROOT, "data_raw"), os.path.join(ROOT, "data_proc")
os.makedirs(OUT, exist_ok=True)
PANEL = ["Daporinad", "Gemcitabine", "THZ-2-102-1", "Bleomycin", "Trametinib", "SNX-2112"]
P_KEEP = 500

P = pd.read_csv(os.path.join(RAW, "data_protein_quantification.txt"), sep="\t", index_col=0)
X = P.select_dtypes("number").T.astype(float)            # lines x proteins
X = X.sort_index()
obs_rate = X.notna().mean(axis=0)
cand = obs_rate[obs_rate >= 0.5].index
med = X[cand].median(axis=0)
mad = (X[cand] - med).abs().median(axis=0) * 1.4826
stats = pd.DataFrame({"protein": cand, "obs_rate": obs_rate[cand].values, "mad": mad.values})
stats = stats.sort_values(["mad", "protein"], ascending=[False, True]).reset_index(drop=True)
keep = stats["protein"].iloc[:P_KEEP].tolist()
stats["selected"] = stats["protein"].isin(keep)
stats.to_csv(os.path.join(OUT, "proteins.csv"), index=False)
X[keep].to_csv(os.path.join(OUT, "X500.csv"))

D = pd.read_csv(os.path.join(RAW, "data_drug_treatment_auc.txt"), sep="\t", index_col=0)
A = D.loc[PANEL, [c for c in D.columns[3:] if c in X.index]].apply(pd.to_numeric, errors="coerce").T
A = A.reindex(X.index)
A.to_csv(os.path.join(OUT, "auc_panel.csv"))
D.loc[PANEL, ["NAME", "DESCRIPTION"]].to_csv(os.path.join(OUT, "drug_annotation.csv"))

S = pd.read_csv(os.path.join(RAW, "data_clinical_sample.txt"), sep="\t", comment="#").set_index("SAMPLE_ID")
L = S.reindex(X.index)[["PROTEOMICS_10PLEX_ID", "PROTEOMICS_TMT_LABEL", "LINEAGE", "CANCER_TYPE"]]
L["row_missing_p500"] = X[keep].isna().mean(axis=1)
L.to_csv(os.path.join(OUT, "lines.csv"))

Xk = X[keep]
print(f"lines={X.shape[0]} proteins_total={X.shape[1]} obs>=50%={len(cand)} kept={len(keep)}")
print(f"missing cells (p=500): {Xk.isna().values.mean():.4f}; row missing range "
      f"{Xk.isna().mean(axis=1).min():.3f}-{Xk.isna().mean(axis=1).max():.3f}; complete proteins "
      f"{int((Xk.notna().all(axis=0)).sum())}")
print("n per drug:", A.notna().sum(axis=0).to_dict())
