import os
import torch
import torch.nn as nn
import rdkit
import rdkit.Chem as Chem
from utils import *
from torch_geometric import data as DATA
from create_dataset import smile_to_graph
processed_data_file_train = 'data/train_data.pt'

train_data = torch.load(processed_data_file_train)

print(train_data[0])

# smiles= 'COc1cc(CCc2cc(-c3nc4ccc(N5CCN(C)CC5)cc4[nH]3)n[nH]2)cc(OC)c1'
# c_size, x, edge_index, edge_weights = smile_to_graph(smiles)

# data = DATA.Data(x=torch.Tensor(x), edge_index=torch.LongTensor(edge_index).transpose(1, 0), edge_weights=torch.FloatTensor(edge_weights),y=torch.FloatTensor(1))
# print(data)