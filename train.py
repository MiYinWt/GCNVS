import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset import *
from model import GCNnet
from utils import *


TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 512
LR = 0.0002
LOG_INTERVAL = 10
NUM_EPOCHS = 1000
cuda_name = 'cuda:0'

def train(model, device, train_loader, optimizer, epoch):
    loss_fn = nn.MSELoss()
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


train_data,train_label = proccesed_data('data/train.csv')
test_data,test_label = proccesed_data('data/test.csv')
val_data,val_label = proccesed_data('data/val.csv')

train_loader = DataLoader(VSDataset(train_data,train_label),batch_size=512, shuffle=False)
test_loader = DataLoader(VSDataset(test_data,test_label), batch_size=512, shuffle=False)

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = GCNnet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_epoch = -1
model_file_name = 'model_finish.model'
result_file_name = 'result.csv'
for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch+1)
    G,P = predicting(model, device, test_loader)
    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
    if ret[1]<best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret)))
        best_epoch = epoch+1
        best_mse = ret[1]
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse)
    else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse)