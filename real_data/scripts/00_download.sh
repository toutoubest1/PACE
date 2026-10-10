#!/bin/sh
# Download the primary dataset (cBioPortal DataHub study ccle_broad_2019) and the
# candidate datasets profiled in REAL_DATA_SELECTION.md. Files are Git-LFS objects;
# their SHA-256 must equal the LFS object ids recorded in data_raw/SHA256SUMS.
set -e
cd "$(dirname "$0")/.."
B=https://media.githubusercontent.com/media/cBioPortal/datahub/master/public
mkdir -p data_raw data_candidates
for f in data_protein_quantification.txt data_drug_treatment_auc.txt data_drug_treatment_ic50.txt \
         data_clinical_sample.txt data_clinical_patient.txt; do
  [ -s data_raw/$f ] || curl -sSfL -o data_raw/$f $B/ccle_broad_2019/$f
done
R=https://raw.githubusercontent.com/cBioPortal/datahub/master/public/ccle_broad_2019
for f in README.md LICENSE meta_study.txt meta_protein_quantification.txt meta_drug_treatment_auc.txt; do
  [ -s data_raw/$f ] || curl -sSfL -o data_raw/$f $R/$f
done
for s in brca_cptac_2020 luad_cptac_2020; do
  for f in data_protein_quantification.txt data_clinical_sample.txt data_clinical_patient.txt; do
    [ -s data_candidates/${s}__$f ] || curl -sSfL -o data_candidates/${s}__$f $B/$s/$f
  done
done
[ -s data_candidates/riboflavin.rda ] || curl -sSfL -o data_candidates/riboflavin.rda https://raw.githubusercontent.com/cran/hdi/master/data/riboflavin.RData
(cd data_raw && sha256sum -c SHA256SUMS)
