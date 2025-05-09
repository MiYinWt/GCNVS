import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset import *
from model import GCNnet
from utils import *



train_data,train_label = proccesed_data('data/train.csv')


train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=2, shuffle=False)

def train(model,device,train_loader,optimizer,criterion):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    return loss.item()

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# model = GCNnet().to(device)


