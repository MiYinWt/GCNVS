import os
import torch
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
    c_size = mol.GetNumAtoms()
    features = []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )


    edge_index = []
    edge_weights = []
    for bond in mol.GetBonds():
        e1 = bond.GetBeginAtomIdx()
        e2 = bond.GetEndAtomIdx()
        edge_index.append([e1, e2])
        edge_weights.append(bond_weight(bond))

    return c_size, features, edge_index , edge_weights


def proccesed_data(data_path):
    df = pd.read_csv(data_path)
    smiles = df['Ligand SMILES'].tolist()
    labels = df['Label'].tolist()
    assert len(smiles) == len(labels), "Number of SMILES and labels must match"
    data_list = []
    for i in range(len(smiles)):
        c_size, features, edge_index , edge_weights = smile_to_graph(smiles[i])
        data = DATA.Data(x=torch.tensor(features, dtype=torch.float), 
                          edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
                          edge_weights=torch.tensor(edge_weights, dtype=torch.float),
                          )
        
        data_list.append(data)

    return data_list,labels
