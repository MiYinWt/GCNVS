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
test_data,test_label = proccesed_data('data/test.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=512, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=512, shuffle=False)


def train(model,device,train_loader,epochs,optimizer):  
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    model.train()
    for i in range(epochs):
        for batch_idx,data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i,
                                                                           batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))



device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


train(GCNnet(),device,train_loader,500,torch.optim.Adam(GCNnet().parameters(),lr=0.002))


