import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import *
from model import GCNnet
from utils import *


train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=32, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=32, shuffle=False)


def train(model,device,train_loader,epochs,optimizer):  
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model = model.to(device)
    model.train()
    for i in range(epochs):
        for batch_idx,data in enumerate(train_loader):
            data = data.to(device)
            
            out = model(data)
            loss = loss_fn(out, data.y.to(torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train epoch: {:5d} /{:5d} [{:3.2f}%]\tLoss: {:.6f}'.format(i,
                                                                        epochs,
                                                                        100. * (i + 1) / epochs,
                                                                        loss.item()))
    torch.save(model, 'model.pkl')

# def test(test_loader,device):
#     model = torch.load('model.pkl')    
#     model.eval()
#     model.to(device)
#     with torch.no_grad():
#         for dada in test_loader:
#             data = dada.to(device)
#             out = model(data)
#             accuracy = torch.max(F.softmax(out), 1)[1]    
#             correct = (accuracy == data.y).sum()
#             print('accuracy:\t %', 100*correct.item()/len(accuracy))




device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


train(GCNnet(),device,train_loader,500,torch.optim.Adam(GCNnet().parameters(),lr=0.01))
#test(test_loader)