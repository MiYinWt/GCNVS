import os
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.transforms import Compose
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import pandas as pd
import networkx as nx

from utils import *

class VSDataset(Dataset):
    def __init__(self,data_list,label):
        self.data_list = data_list
        self.label =label
        print(f"dataset loaded with {len(self.data_list)} samples.")

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        data = self.data_list[idx]
        label = self.label[idx]
        data.y = torch.tensor(label, dtype=torch.long)
        return data


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    features = []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
        # features.append( feature )

    edge_index = []
    edge_weights = []
    for bond in mol.GetBonds():
        e1 = bond.GetBeginAtomIdx()
        e2 = bond.GetEndAtomIdx()
        edge_index.append([e1, e2])
        edge_index.append([e2, e1])
        edge_weights.append(bond_weight(bond))
        edge_weights.append(bond_weight(bond))
    return  features, edge_index , edge_weights


def proccesed_data(data_path):
    df = pd.read_csv(data_path)
    smiles = df['Ligand SMILES'].tolist()
    labels = df['Label'].tolist()
    ##   Number of SMILES and labels must match
    assert len(smiles) == len(labels)
    data_list = []
    for i in range(len(smiles)):
        features, edge_index , edge_weights = smile_to_graph(smiles[i])
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights=torch.tensor(edge_weights, dtype=torch.float)
        edge_index,edge_weights = add_self_loops(edge_index,edge_weights)
        data = DATA.Data(x=torch.tensor(features, dtype=torch.float), 
                          edge_index=edge_index, 
                          edge_weights=edge_weights,
                          )
        
        data_list.append(data)

    return data_list,labels
