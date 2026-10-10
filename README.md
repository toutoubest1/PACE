# PACE

Python code for pattern-ancillary robust estimating equations in sparse regression with missing and contaminated covariates.

The public estimator is `pace.fit_pace`, which uses five-fold cross-fitting (PACE-CF5). The repository includes simulation generators, comparator implementations, experiment runners, and scripts for the CCLE proteomics/drug-response application.

## Installation

Create a Python environment and install the dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The dependency versions are retained from the supplied research repository. They have not been revalidated in a fresh environment during this packaging pass.

## Estimator

```python
from pace import fit_pace

beta_hat = fit_pace(Xobs, y, M)
```

`Xobs` is an n-by-p array, `y` is a length-n response, and `M` is a Boolean observation mask. Missing cells may contain zero; the mask identifies which entries are observed.

## Layout

- `pace/`: estimator, moment construction, weights, solver and comparators.
- `sim/`: simulation generators and method registry.
- `experiments/`: simulation settings and runners.
- `analysis/`: summaries, tables and figures generated from local simulation outputs.
- `real_data/scripts/`: download, preprocessing, fitting and analysis.
- `real_data/data_raw/SHA256SUMS`: expected input-file checksums; no observations or results.

## Running the analyses

```sh
bash experiments/run_all_final.sh
sh real_data/scripts/run_all.sh
```

To run both workflows and the manuscript-output scripts:

```sh
sh reproduce.sh
```

These commands run substantial computations and the real-data workflow downloads public data. A shell with `curl`, `sha256sum`, `seq` and `xargs` is required for the download/workflow scripts.

Generated logs, results, tables, figures and downloaded/preprocessed datasets are excluded from version control. This distribution contains no stored experiment results. Table-generation scripts require outputs from a prior local run; baseline-comparison checks against archived results are not included.

## Data and comparators

Real-data download scripts use the cBioPortal DataHub `ccle_broad_2019` study and candidate datasets. Source-provided data licenses apply to downloaded datasets. The input checksums are retained; upstream changes may cause verification to fail.

Sparse LTS, Huber-Lasso, Filter-MICE-LTS and IPW-LPD-Lasso are Python reimplementations for this study, rather than reference R implementations. The radial-weighting baseline retains diagnostic options for earlier implementation defects; corrected settings are the default.

