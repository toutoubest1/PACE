"""One train/test split for one drug, all methods (protocol Sections 3-6).
Usage: python scripts/03_run_split.py DRUG REP [debug]
debug replaces the response by N(0,1) noise (seed 999) """
import os, sys, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, METHODS, load, drug_data, split, standardiser, to_input, mice_test_imputer,
                    predict, metrics)

drug, rep = sys.argv[1], int(sys.argv[2])
debug = "--debug" in sys.argv
odir = os.path.join(ROOT, "results", "debug" if debug else "raw")
os.makedirs(odir, exist_ok=True)
tag = f"{drug}_r{rep:02d}"
fcsv = os.path.join(odir, tag + ".csv")
if os.path.exists(fcsv):
    print("exists", fcsv); sys.exit(0)

X, A, L = load()
Xnan, y, lines = drug_data(X, A, drug)
if debug:
    y = np.random.default_rng(999).standard_normal(len(y))
tr, te = split(len(y), rep)
med, sc = standardiser(Xnan[tr])
Ztr, Mtr = to_input(Xnan[tr], med, sc)
Zte, _ = to_input(Xnan[te], med, sc)
imp = mice_test_imputer(Xnan[tr])
Zte_mice = (imp.transform(Xnan[te]) - med) / sc
ymed = np.median(y[tr]); ytr_c = y[tr] - ymed

rows, coefs = [], {}
for name, fn in METHODS.items():
    t0 = time.process_time()
    b = np.asarray(fn(Ztr, ytr_c, Mtr), float)
    t = time.process_time() - t0
    coefs[name] = b
    m1 = metrics(y[te], ymed + predict(b, Ztr, ytr_c, Zte))
    m2 = metrics(y[te], ymed + predict(b, Ztr, ytr_c, Zte_mice))
    rows.append(dict(Method=name, drug=drug, rep=rep, n_train=len(tr), n_test=len(te), Size=int(np.sum(b != 0)),
                     Time=t, **m1, **{k + "_mice": v for k, v in m2.items()}))
null = metrics(y[te], np.full(len(te), ymed))
rows.append(dict(Method="Null", drug=drug, rep=rep, n_train=len(tr), n_test=len(te), Size=0, Time=0.0,
                 **null, **{k + "_mice": v for k, v in null.items()}))
df = pd.DataFrame(rows)
for s in ["", "_mice"]:
    df["relMAE" + s] = df["MAE" + s] / null["MAE"]
np.savez_compressed(os.path.join(odir, tag + "_coef.npz"), **{k.replace(" ", "_"): v for k, v in coefs.items()},
                    test_idx=te)
df.to_csv(fcsv, index=False)
print(tag, "done", round(df.Time.sum(), 1), "s")
