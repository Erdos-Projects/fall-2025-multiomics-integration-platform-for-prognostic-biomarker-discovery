import pandas as pd
from . import paths

def printDataShape(df):
    if 'name' in df.attrs:
        print(f'{df.attrs['name']} dataframe has shape {df.shape}')
    else:
        print('This dataframe has shape {df.shape}')

def loadTCGAData():
    rppa_raw = pd.read_csv(paths.getRPPAPath(), sep='\t', index_col = 0) 
    rppa_raw.attrs['name'] = 'RPPA'

    rna_raw = pd.read_csv(paths.getRNAPath(), sep='\t',low_memory=False)
    rna_raw.attrs['name'] = 'RNA-Seq'

    clin_raw = pd.read_csv(paths.getClinicalPath(), sep='\t', comment="#")
    clin_raw.attrs['name'] = 'Clinical' 

    return rppa_raw, rna_raw, clin_raw

def loadEGFRvIIIData():
    egfrviii = pd.read_csv(paths.getEGFRPath())
    egfrviii.attrs['name'] = 'EGFRvIII'

    return egfrviii