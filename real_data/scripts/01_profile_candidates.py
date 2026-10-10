"""Profile the candidate datasets (dimensions and natural missingness) for REAL_DATA_SELECTION.md.
Uses predictors and response availability only; no model is fitted.
"""
import json, os, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, CAND = os.path.join(ROOT, "data_raw"), os.path.join(ROOT, "data_candidates")
out = {}


def miss_profile(X):
    m = X.isna()
    cm, rm = m.mean(axis=0), m.mean(axis=1)
    q = [0, .1, .25, .5, .75, .9, 1]
    return dict(n=int(X.shape[0]), p=int(X.shape[1]), cell_missing=round(float(m.values.mean()), 4),
                col_missing_quantiles=dict(zip(map(str, q), np.round(cm.quantile(q).values, 3).tolist())),
                row_missing_quantiles=dict(zip(map(str, q), np.round(rm.quantile(q).values, 3).tolist())),
                complete_columns=int((cm == 0).sum()), columns_le_50pct_missing=int((cm <= .5).sum()),
                distinct_row_patterns=int(m.drop_duplicates().shape[0]))


#primary candidate: CCLE proteomics (Nusinow et al. 2020) via cBioPortal ccle_broad_2019
P = pd.read_csv(os.path.join(RAW, "data_protein_quantification.txt"), sep="\t", index_col=0)
X = P.select_dtypes("number").T
prof = miss_profile(X)
S = pd.read_csv(os.path.join(RAW, "data_clinical_sample.txt"), sep="\t", comment="#").set_index("SAMPLE_ID")
plex = S.loc[X.index, "PROTEOMICS_10PLEX_ID"]
m = X.isna()
allm = m.groupby(plex.values).all()
sizes = plex.value_counts()
block_cells = sum(int(allm.loc[k].sum()) * int(sizes[k]) for k in allm.index)
prof["n_plexes"] = int(plex.nunique())
prof["share_missing_cells_whole_plex"] = round(block_cells / int(m.values.sum()), 4)
prof["row_missing_by_plex_range"] = np.round(m.mean(axis=1).groupby(plex.values).mean().agg(["min", "max"]).values, 3).tolist()
D = pd.read_csv(os.path.join(RAW, "data_drug_treatment_auc.txt"), sep="\t", index_col=0)
lines = [c for c in D.columns[3:] if c in X.index]
A = D[lines].apply(pd.to_numeric, errors="coerce")
nA = A.notna().sum(axis=1)
prof["drug_auc_entries"] = int(D.shape[0])
prof["lines_with_proteomics_and_any_auc"] = len(lines)
prof["drugs_with_n_ge_250"] = int((nA >= 250).sum())
prof["auc_n_per_drug_quantiles"] = np.round(nA.quantile([0, .25, .5, .75, 1]).values, 0).tolist()
out["ccle_proteomics_gdsc_auc"] = prof

# CPTAC tumour proteomics (cBioPortal)
for s in ["brca_cptac_2020", "luad_cptac_2020"]:
    P = pd.read_csv(os.path.join(CAND, f"{s}__data_protein_quantification.txt"), sep="\t", index_col=0)
    Xs = P.select_dtypes("number").T
    pr = miss_profile(Xs)
    C = pd.read_csv(os.path.join(CAND, f"{s}__data_clinical_patient.txt"), sep="\t", comment="#")
    num = C.apply(pd.to_numeric, errors="coerce")
    pr["numeric_patient_attributes"] = {c: int(num[c].notna().sum()) for c in num.columns if num[c].notna().sum() > 0}
    out[s] = pr

# riboflavin (hdi): dimensions only 
try:
    import rdata
    r = rdata.parser.parse_file(os.path.join(CAND, "riboflavin.rda"))
    x = r.object.value[0].value[1]  # AsIs matrix column 'x'
    dim = [int(v) for v in x.attributes.value[0].value] if hasattr(x, "attributes") else None
    out["riboflavin"] = dict(dim=dim)
except Exception as e:  # dimension check is informational only
    out["riboflavin"] = dict(note=f"not parsed ({type(e).__name__}); documented n=71, p=4088, no NA")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "selection_profile.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
