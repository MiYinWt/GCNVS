import torch
from torch_geometric.loader import DataLoader
from model import GCNnet
from dataset import *


model = torch.load('model_finish.model')
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = GCNnet()
model.load_state_dict(torch.load('model_finish.model'))

model.to(device)


val_data,val_label = proccesed_data('data/train.csv')

# val_loader = DataLoader(VSDataset(val_data,val_label))

with torch.no_grad():
    for i in range(len(val_data)):
        out = model(val_data[i].to(device))
        print(val_label[i], out)