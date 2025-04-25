import pandas as pd

df = pd.read_csv('./data/BindingDB_All.tsv',sep='\t', low_memory=False)

smiles = 'Ligand SMILES'
TargetName = 'Target Name'
IC='IC50 (nM)'

