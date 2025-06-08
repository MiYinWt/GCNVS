import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.patheffects as pe
data_path = 'data/total.csv'
# 示例数据（假设从数据库获取的 SMILES 及其活性标签）
df = pd.read_csv(data_path)
smiles_list = df['Ligand SMILES'].tolist()
labels_list = df['Label'].tolist()
##   Number of SMILES and labels must match
assert len(smiles_list) == len(labels_list)

mol_objects = [Chem.MolFromSmiles(smile) for smile in smiles_list]
mol_weights_active = []
mol_weights_inactive = []

for i, mol in enumerate(mol_objects):
    if labels_list[i]:
        mol_weights_active.append(Descriptors.MolWt(mol))
    else:
        mol_weights_inactive.append(Descriptors.MolWt(mol))


