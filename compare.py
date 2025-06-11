from sklearn.naive_bayes import BernoulliNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  roc_auc_score,f1_score,precision_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

def get_fingerprints(smiles_list):
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.array(fp)
            fingerprints.append(fp_array)
        else:
            raise ValueError(f"Invalid SMILES string: {smi}")
    return np.array(fingerprints)

scaler = MinMaxScaler()
df_train = pd.read_csv('data/train.csv')
Smiles_train = df_train['Ligand SMILES'].tolist()
y_train = df_train['Label'].tolist()

df_test = pd.read_csv('data/test.csv')
Smiles_test = df_test['Ligand SMILES'].tolist()
y_test = df_test['Label'].tolist()

X_train = scaler.fit_transform(get_fingerprints(Smiles_train))
X_test = scaler.fit_transform(get_fingerprints(Smiles_test))

# model = BernoulliNB()
# model = LogisticRegression(max_iter=500)
# model = SVC(probability=True)
# model = RandomForestClassifier(n_estimators=100)
model = MLPClassifier(hidden_layer_sizes=(2048, 1024), max_iter=500, random_state=42)
# model = KNeighborsClassifier(n_neighbors=10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.4f}')

precision_score = precision_score(y_test, y_pred)
print(f'Precision Score: {precision_score:.4f}')

y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC: {roc_auc:.4f}')