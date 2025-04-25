import pandas as pd

df = pd.read_csv('./data/BindingDB_All.tsv',sep='\t', low_memory=False)

smiles_col = 'Ligand SMILES'
TargetName_col = 'Target Name'
IC50_col = 'IC50 (nM)'

#Filter Targets

df_filtered = df[df[TargetName_col].str.contains('Fibroblast Growth Factor Receptor 1')]
