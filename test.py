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

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=512, shuffle=True)

for batch_idx,data in enumerate(train_loader):
    if batch_idx % 2 == 0:
        print(data.y.view(-1, 1).float())