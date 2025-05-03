import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.transforms import Compose
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import pandas as pd
import networkx as nx

from utils import *

class VSDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)



    def __len__(self):
        return len(self.df)
    
    def process(self):
        data = self.df

        return super().process()


def smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

# convert dataset to Pytorch Geometric DataLoader
