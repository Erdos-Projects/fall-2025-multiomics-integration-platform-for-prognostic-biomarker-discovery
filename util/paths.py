from pathlib import Path


DATA_FOLDER = Path(__file__).resolve().parents[1].joinpath('data')

def getRPPAPath():
    return DATA_FOLDER.joinpath('TCGA2021').joinpath('data_rppa.txt')

def getRNAPath():
    return DATA_FOLDER.joinpath('TCGA2021').joinpath('data_mrna_seq_v2_rsem.txt')

def getClinicalPath():
    return DATA_FOLDER.joinpath('TCGA2021').joinpath('data_clinical_patient.txt')

def getEGFRPath():
    return DATA_FOLDER.joinpath('Hoogstrate2022').joinpath('The_EGFRvIII_Transcriptome_in_glioblastoma_data_public_Zenodo.csv')