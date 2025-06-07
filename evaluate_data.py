import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
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

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist([mol_weights_active, mol_weights_inactive], bins=10, label=['Active', 'Inactive'], color=['green', 'orange'])
plt.xlabel('MolWt')
plt.ylabel('Count')
plt.title('Molecular Weight Distribution')
plt.legend()
plt.show()