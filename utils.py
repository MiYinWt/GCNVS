from math import sqrt
from scipy import stats
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None

def bond_weight(bond):
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return 1.0
    elif bond_type == Chem.BondType.DOUBLE:
        return 2.0
    elif bond_type == Chem.BondType.TRIPLE:
        return 3.0
    elif bond_type == Chem.BondType.AROMATIC:
        return 1.5
    else:
        return 0.0   #unknown bond type


def atom_features(atom):
    return np.array(one_of_k_encoding(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetTotalValence(), [ 1, 2, 3, 4, 5, 6, 7]) +
                    one_of_k_encoding(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2, 
                          Chem.rdchem.HybridizationType.SP3]) + 
                    one_of_k_encoding(atom.IsInRing(),[True,False])+
                    one_of_k_encoding(atom.GetIsAromatic(),[True,False]))
                    
                    
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowSable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
