import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.transforms import Compose
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import pandas as pd
import networkx as nx

from utils import *

class VSDataset(Dataset):
    def __init__(self, csv_path, processed_dir, dataset_type):
        self.csv_path = csv_path
        super(VSDataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.processed_dir = processed_dir
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.df)
    
    def process(self):
        data = self.df
        smile_list = data['smiles'].tolist()
        y_list = data['Label'].tolist()
        ## print(f"smile_list length: {len(smile_list)}")
        assert len(smile_list) == len(y_list), "smile and y list length must equal"
        data_list = []
        for i in range(len(smile_list)):
            ## print(smile_list[i])
            x , edge_index, edge_weights = smile_to_graph(smile_list[i])
            y = y_list[i]
            data = DATA.Data(x=x, edge_index=edge_index, edge_weights=edge_weights, y=y)
            data_list.append(data)
        torch.save(data_list, os.path.join(self.processed_dir, f'{self.dataset_type}_data.pt'))



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
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

    return features, edge_index , edge_weights

# convert dataset to Pytorch Geometric DataLoader
print("Loading dataset in pytorch geometric format...")

train_dataset = VSDataset(csv_path='./data/train.csv', processed_dir='data/', dataset_type='train')
train_dataset.process()

test_dataset = VSDataset(csv_path='./data/test.csv', processed_dir='data/', dataset_type='test')
test_dataset.process()

val_dataset = VSDataset(csv_path='./data/val.csv', processed_dir='data/', dataset_type='val')
val_dataset.process()

print("All Dataset loaded.")