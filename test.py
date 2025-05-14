import os
import torch
import torch.nn as nn
import rdkit
import rdkit.Chem as Chem
from torch_geometric.loader import DataLoader
from utils import *
from torch_geometric import data as DATA
from dataset import *

train_data,train_label = proccesed_data('data/train.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),shuffle=False)
for data in train_loader:
    print("data:\n",data.edge_index,data.edge_weights)
    break