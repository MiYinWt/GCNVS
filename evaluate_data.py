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
tPSA_active = []
tPSA_inactive = []
QED_active = []
QED_inactive = []
ring_active =[]
ring_inactive = []

for i, mol in enumerate(mol_objects):
    if labels_list[i]:
        mol_weights_active.append(Descriptors.MolWt(mol))
        tPSA_active.append(Descriptors.TPSA(mol))
        QED_active.append(Descriptors.qed(mol))
        ring_active.append(Descriptors.RingCount(mol))
    else:
        mol_weights_inactive.append(Descriptors.MolWt(mol))
        tPSA_inactive.append(Descriptors.TPSA(mol))
        QED_inactive.append(Descriptors.qed(mol))
        ring_inactive.append(Descriptors.RingCount(mol))


df_out = pd.DataFrame({
    'MolWt_active': mol_weights_active,
    'tPSA_active': tPSA_active,
    'QED_active': QED_active,
    'RingCount_active': ring_active
})
edg_out = pd.DataFrame({
    'MolWt_inactive': mol_weights_inactive
    ,'tPSA_inactive': tPSA_inactive,
    'QED_inactive': QED_inactive,
    'RingCount_inactive': ring_inactive
})

df_out.to_csv('eval_active.csv', index=False)
edg_out.to_csv('eval_inactive.csv', index=False)