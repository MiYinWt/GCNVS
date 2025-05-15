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
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    model = model.to(device)
    model.train()

    for batch_idx,data in enumerate(train_loader):
        data = data.to(device)
        out = model(data)
        # print("out:\n",out)
        loss = loss_fn(out, data.y.to(torch.long).to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    print('Train epoch: {:5d} /{:5d} [{:3.2f}%]\tLoss: {:.6f}'.format(epoch,
                                                                    NUM_EPOCHS,
                                                                    100. * (epoch) / NUM_EPOCHS,
                                                                    loss.item()))


# def test(model,device,test_loader):
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

NUM_EPOCHS = 500

train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=64, shuffle=True)
test_loader = DataLoader(VSDataset(test_data,test_label),batch_size=32, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = GCNnet()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for i in range(NUM_EPOCHS):
    train(model, device, train_loader, i+1, optimizer)
    scheduler.step()