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

def train(model,device,train_loader,epoch,optimizer):  
    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = torch.nn.BCELoss()
    model = model.to(device)
    model.train()

    for batch_idx,data in enumerate(train_loader):
        data = data.to(device)
        out = model(data)
        # print("out:\n",out)
        loss = loss_fn(out, data.y.to(torch.float).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    print('Train epoch: {:5d} /{:5d} [{:3.2f}%]\tLoss: {:.6f}'.format(epoch,
                                                                    NUM_EPOCHS,
                                                                    100. * (epoch) / NUM_EPOCHS,
                                                                    loss.item()))


# def test(model, device, test_loader):
#     model.eval()  
#     model.to(device)
#     correct = 0
#     total = 0
#     with torch.no_grad():  
#         for data in test_loader:
#             data = data.to(device)
#             out = model(data)
#             print("out:\n",out)
#             pred = torch.argmax(out, dim=1)
#             print("pred:\n",pred)  
#             correct += (pred == data.y).sum().item()
#             total += data.y.size(0)
#     accuracy = 100. * correct / total
#     print(f'Test Accuracy: {accuracy:.2f}%')




NUM_EPOCHS = 500

train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')
val_data,val_label = proccesed_data('data/val.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=64, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=64, shuffle=False)
val_loader = DataLoader(VSDataset(val_data,val_label),batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = GCNnet()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)


for i in range(NUM_EPOCHS):
    train(model, device, train_loader, i+1, optimizer)


# test(model, device, test_loader)