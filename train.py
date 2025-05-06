import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import GCNnet
from utils import *

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0002
LOG_INTERVAL = 10
NUM_EPOCHS = 1000
cuda_name = 'cuda:0'

def train(model, device, train_loader, optimizer, epoch):
    loss_fn = nn.MSELoss()
    print('Training on {} samples...'.format(len(train_loader)))
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
                                                                           len(train_loader),
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


processed_data_file_train = 'data/train_data.pt'
processed_data_file_test = 'data/test_data.pt'


if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    print('please prepare data in pytorch format!')
else:
    train_data = torch.load(processed_data_file_train)
    test_data = torch.load(processed_data_file_test)
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # training the model
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