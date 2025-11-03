import pandas as pd

def tcga_base_id(x):
    # keep first three parts: TCGA-XX-YYYY
    return '-'.join(x.split('-')[:3])

def cleanTCGAData(rppa_raw, rna_raw, clin_raw):
    # Clean RPPA columns
    rppa = rppa_raw.copy()
    rppa.columns = [tcga_base_id(c) for c in rppa.columns]

    # Clean RNA columns (skip the first two metadata columns)
    rna = rna_raw.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id']).copy()
    rna.columns = [tcga_base_id(c) for c in rna.columns]

    # Clean clinical IDs
    clin = clin_raw.copy()
    clin['PATIENT_ID'] = clin['PATIENT_ID'].apply(tcga_base_id)

    # Remove duplicated patient IDs and take transpose for rppa and rna
    rppa = rppa.loc[:, ~rppa.columns.duplicated()]
    rppa = rppa.T

    rna  = rna.loc[:,  ~rna.columns.duplicated()]
    rna  = rna.T

    clin = clin.drop_duplicates(subset='PATIENT_ID', keep='first')

    # Convert everything to numeric
    rppa = rppa.apply(pd.to_numeric, errors='coerce')
    rna  = rna.apply(pd.to_numeric, errors='coerce')

    # Drop features (columns) with >20% missing values
    rppa = rppa.dropna(axis=1, thresh=int(0.8 * len(rppa)))
    rna  = rna.dropna(axis=1, thresh=int(0.8 * len(rna)))

    # Fill remaining NaNs with feature means
    rppa = rppa.fillna(rppa.mean())
    rna  = rna.fillna(rna.mean())

    return rppa, rna, clin